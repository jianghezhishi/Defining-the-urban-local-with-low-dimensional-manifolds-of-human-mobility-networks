'''
选址流形均匀性检验代码整理
'''
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import csv
from shapely.geometry import Polygon, Point, box
import alphashape
from shapely.geometry import Polygon, MultiPolygon
import networkx as nx
import ast
from shapely import wkt
import os
import warnings
import matplotlib
from scipy.spatial import KDTree, ConvexHull, Delaunay
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
def cid_to_xy(cid,ynum=40):
    xid=int((cid-1)/ynum)
    yid=cid-1-xid*ynum
    return xid,yid

def find_largest_connected_component(edges):
    # 创建一个无向图
    G = nx.Graph()

    # 将边添加到图中
    G.add_edges_from(edges)

    # 查找图中所有的连通组件
    connected_components = list(nx.connected_components(G))

    # 找到最大连通组件
    largest_component = max(connected_components, key=len)

    # 将最大连通组件的节点返回
    return list(largest_component)
def write_grid_geo_minpop10(file,minlam=5,minpop=10,title=r'E:\流形研究\20260201名古屋选址流形\minpop10_alphashape\grid.csv'):
    cid_cover = {}  # 即格子覆盖人数，用minpop筛选
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            if len(eval(row[1])) >= minpop:
                cid_cover[int(row[0])] = eval(row[1])
    print('cover read')

    cp_lam = {}
    nodes = list(cid_cover.keys())
    for i in range(len(nodes) - 1):
        if i % 100 == 0:
            print(i)
        cid = nodes[i]
        for j in range(i + 1, len(nodes)):
            cid1 = nodes[j]
            lam = len(cid_cover[cid] & cid_cover[cid1])
            if lam >= minlam:
                cp_lam[(cid, cid1)] = lam
    print('lam ready')
    nodes = find_largest_connected_component(list(cp_lam.keys()))
    with open(title, 'w', newline='') as g:
        wt = csv.writer(g)
        for cid in nodes:
            x, y = cid_to_xy(cid)
            wt.writerow([cid, x, y])
    return
def load_selection(npy_path):
    sel = np.load(npy_path)
    return set(sel.tolist())
def square_from_center(x, y, size=1.0):
    h = size / 2
    return Polygon([
        (x-h, y-h),
        (x+h, y-h),
        (x+h, y+h),
        (x-h, y+h)
    ])
def read_points_csv(path):
    return pd.read_csv(
        path,
        header=None,
        names=["id", "x", "y"]
    )
def output_grid_cells(grid_csv, npy_path, out_csv):
    grid = read_points_csv(grid_csv)
    sel = load_selection(npy_path)

    records = []
    for _, r in grid.iterrows():
        poly = square_from_center(r.x, r.y, 1.0)
        records.append({
            "id": r.id,
            "x": r.x,
            "y": r.y,
            "wkt": poly.wkt,
            "selected": r.id in sel
        })

    pd.DataFrame(records).to_csv(out_csv, index=False)



def read_cover_csv(path):
    """
    读取 cover csv：
    - 无表头
    - 第一列：id
    - 第二列：集合（字符串表示）
    返回 DataFrame: id, population
    """
    df = pd.read_csv(path, header=None)

    if df.shape[1] < 2:
        raise ValueError("cover csv 至少需要两列：id 和 集合")

    df.columns = ["id", "cover_set"]

    def parse_len(x):
        if pd.isna(x):
            return 0
        if isinstance(x, (list, set)):
            return len(x)
        try:
            return len(ast.literal_eval(x))
        except Exception:
            return 0

    df["population"] = df["cover_set"].apply(parse_len)

    return df[["id", "population"]]


def read_manifold(file, nc=2, opt=1):
    res, nodes = [], []
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            temp = row[1][1:-1]
            try:
                cid = int(row[0])#这里需要解决，有些数据的cid是字符串，有些是int的问题，因为读取矩阵时使用eval读取nodes，因此当cid是int时需要在这里也按int读取
            except:
                cid=row[0]
            else:
                pass
            if opt == 1:
                xy = []
                flag = 0
                flag1=0
                for it in temp.split(' '):
                    try:
                        num = float(it)
                    except:
                        pass
                    else:
                        flag += 1
                        xy.append(num)
                        if abs(num)<=0.01:
                            flag1+=1
                if flag == nc:
                    if flag1<nc:
                        nodes.append(cid)
                        res.append(xy)
                    else:
                        print(xy)
                else:
                    print('err')
            else:
                nodes.append(cid)
                res.append(eval(row[1]))
    return res, nodes
def read_res_with_manifold(res_csv):
    """
    使用你提供的 read_manifold 读取 res
    返回：
      X: ndarray (N, 2)
      nodes: list
    """
    res, nodes = read_manifold(res_csv, nc=2, opt=1)
    X = np.array(res, dtype=float)
    return X, nodes


def alpha_shape_from_points(X, alpha):
    """
    从点云 X (N,2) 计算 alpha shape
    """
    hull = alphashape.alphashape(X, alpha)

    # 兜底：确保是 Polygon / MultiPolygon
    if isinstance(hull, (Polygon, MultiPolygon)):
        return hull
    else:
        print('no concave')
        return hull.convex_hull




def generate_res_voronoi_csv(
    res_csv,
    site_npy,
    out_csv,
    alpha=0.2
):
    """
    （3）res → 泰森多边形（alpha shape 裁剪）→ CSV

    输出列：
      id, x, y, wkt, selected
    """

    # 1. 读取 res
    X, nodes = read_res_with_manifold(res_csv)

    # 2. 读取选址点
    site_ids = set(np.load(site_npy).tolist())

    # 3. 点 GeoDataFrame
    point_gdf = gpd.GeoDataFrame(
        {
            "id": nodes,
            "x": X[:, 0],
            "y": X[:, 1],
            "geometry": gpd.points_from_xy(X[:, 0], X[:, 1])
        },
        #crs="EPSG:3857"
    )

    # 4. 整体 alpha shape（作为 Voronoi 边界）
    boundary = alpha_shape_from_points(X, alpha=alpha)
    print(boundary)
    # 5. 生成泰森多边形（整体）
    vor_geom = point_gdf.geometry.voronoi_polygons()
    vor_gdf = gpd.GeoDataFrame(
        {"geometry": vor_geom},
        #crs=point_gdf.crs
    )

    # 6. 裁剪到 alpha shape
    vor_gdf = vor_gdf.clip(boundary)

    # 7. 空间连接，绑定 id
    vor_gdf = gpd.sjoin(
        vor_gdf,
        point_gdf,
        predicate="intersects",
        how="left"
    )

    # 8. 输出 CSV（每点一个泰森多边形）
    rows = []
    for _, r in vor_gdf.iterrows():
        rows.append({
            "id": r["id"],
            "x": r["x"],
            "y": r["y"],
            "wkt": r.geometry.wkt,
            "selected": (r["id"] in site_ids)
        })

    out_df = pd.DataFrame(
        rows,
        columns=["id", "x", "y", "wkt", "selected"]
    )

    out_df.to_csv(out_csv, index=False)
    return
def plot_alphashape_results(csv_path, title):
    df = pd.read_csv(csv_path)

    # polygon 只作为范围
    gdf_poly = gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkt(df["wkt"]),
        #crs="EPSG:3857"
    )

    # 点用于标注选址
    if "cx" in df.columns:
        px, py = df["cx"], df["cy"]
    else:
        px, py = df["x"], df["y"]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1️⃣ 画 polygon（范围，不区分 T/F）
    gdf_poly.plot(
        ax=ax,
        color="lightgrey",
        alpha=0.25,
        edgecolor="none"
    )

    # 2️⃣ 非选址点（F）
    mask_F = ~df["selected"]
    ax.scatter(
        px[mask_F],
        py[mask_F],
        s=10,
        c="steelblue",
        alpha=0.35,
        linewidths=0
    )

    # 3️⃣ 选址点（T）
    mask_T = df["selected"]
    ax.scatter(
        px[mask_T],
        py[mask_T],
        s=60,
        c="darkred",
        alpha=0.9,
        zorder=5
    )

    #ax.set_title(title)
    ax.axis("off")
    plt.savefig(title,dpi=300)
    plt.show()
    return

def merge_pol(file,output_dir):
    # 读取数据

    df = pd.read_csv(file)

    df['geometry'] = df['wkt'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

    # 合并所有多边形
    merged = gdf.unary_union

    # 保存结果为shapefile
    result_gdf = gpd.GeoDataFrame(geometry=[merged], crs='EPSG:4326')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存为shapefile（注意：shapefile需要多个文件，这里会生成多个文件）
    shp_output_path = os.path.join(output_dir, 'manif.shp')
    result_gdf.to_file(shp_output_path, driver='ESRI Shapefile')

    print(f"合并完成！保存为 {shp_output_path}")
    print("Shapefile包含以下文件：")
    for file in os.listdir(output_dir):
        if file.startswith('merged_result'):
            print(f"  - {file}")

    # 可视化部分
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 显示原始多边形
    gdf.plot(ax=axes[0],
             alpha=0.5,
             edgecolor='red',
             facecolor='lightblue',
             linewidth=1.5)
    axes[0].set_title(f'原始多边形 (共{len(gdf)}个)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('经度', fontsize=10)
    axes[0].set_ylabel('纬度', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. 显示合并后的多边形（从shapefile重新读取确保正确）
    merged_gdf = gpd.read_file(shp_output_path)
    merged_gdf.plot(ax=axes[1],
                    alpha=0.7,
                    edgecolor='blue',
                    facecolor='lightgreen',
                    linewidth=2)
    axes[1].set_title('合并后的图形 (已保存为shapefile)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('经度', fontsize=10)
    axes[1].set_ylabel('纬度', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 添加总标题
    plt.suptitle('多边形合并前后对比', fontsize=14, fontweight='bold')

    # 调整布局并显示
    plt.tight_layout()

    # 保存可视化图片
    plt.savefig(os.path.join(output_dir, 'merge_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 如果需要更详细的合并信息
    print("\n合并信息:")
    print(f"原始图形数量: {len(gdf)}")
    print(f"合并后图形类型: {type(merged)}")

    # 如果合并后是MultiPolygon，显示有多少个子多边形
    if hasattr(merged, 'geoms'):
        print(f"合并后包含 {len(merged.geoms)} 个子多边形")

    # 显示shapefile属性信息
    print(f"\nShapefile属性表:")
    print(merged_gdf)
    return

def read_uniformity_data(csv_file):
    """
    读取均匀性评估数据CSV文件

    参数:
        csv_file: CSV文件路径，包含x, y, selected字段

    返回:
        DataFrame: 包含所有点的数据
    """
    print(f"读取评估数据文件: {csv_file}")

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 检查必要的列
    required_columns = ['x', 'y', 'selected']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"CSV文件缺少必要列: {missing_columns}")

    # 转换selected字段为布尔值（处理字符串形式的TRUE/FALSE）
    if df['selected'].dtype == 'object':
        df['selected'] = df['selected'].astype(str).str.upper().isin(['TRUE', '1', 'YES', 'T'])

    print(f"数据读取完成:")
    print(f"  总点数: {len(df)}")
    print(f"  选址点数: {df['selected'].sum()}")
    print(f"  X范围: [{df['x'].min():.4f}, {df['x'].max():.4f}]")
    print(f"  Y范围: [{df['y'].min():.4f}, {df['y'].max():.4f}]")

    return df


class PointSetUniformityEvaluator:
    """
    点集均匀性评估系统
    整合数据处理和均匀性评估功能
    """

    def __init__(self, data_csv=None, boundary_shp=None,
                 k_neighbors=4, random_seed=42, boundary_method='shp',
                 x_col='x', y_col='y', use_shp_boundary=True):
        """
        初始化评估器

        参数:
            data_csv: 数据CSV文件路径（包含x, y, selected字段）
            boundary_shp: shp边界文件路径（可选）
            k_neighbors: 考虑的最近邻数量
            random_seed: 随机种子
            boundary_method: 边界提取方法 ('shp')
            x_col: 要使用的x坐标列名
            y_col: 要使用的y坐标列名
            use_shp_boundary: 是否使用shp文件作为边界（如果提供了boundary_shp）
        """
        self.data_csv = data_csv
        self.boundary_shp = boundary_shp
        self.k = k_neighbors
        self.random_seed = random_seed
        self.boundary_method = boundary_method
        self.x_col = x_col
        self.y_col = y_col
        self.use_shp_boundary = use_shp_boundary

        np.random.seed(random_seed)

        self.data = None  # 原始数据DataFrame
        self.points = None  # 要评估的点集（选址点）
        self.boundary_points_raw = None  # 原始的边界点集（所有点）
        self.boundary_polygon = None  # 提取的边界多边形
        self.boundary_points_final = None  # 最终的边界点
        self.boundary_geometry = None  # 从shp读取的几何对象
        self.boundary_crs = None  # shp文件的坐标系
        self.area = None
        self.M = 0  # 要评估的点数

        # 蒙特卡洛模拟相关变量
        self.benchmark_mean = None  # 随机点集Q值的均值
        self.benchmark_std = None  # 随机点集Q值的标准差
        self.benchmark_percentiles = None  # 随机点集Q值的百分位数
        self.Q_distribution = None  # 所有随机点集的Q值分布
        self.num_simulations = 0  # 模拟次数

    def load_and_process_data(self):
        """
        加载并处理CSV数据
        """
        print("=" * 60)
        print("开始加载和处理数据...")

        if self.data_csv is None:
            raise ValueError("未提供数据CSV文件路径")

        # 1. 读取CSV数据
        self.data = read_uniformity_data(self.data_csv)

        # 2. 提取选址点
        selected_data = self.data[self.data['selected'] == True].copy()

        if len(selected_data) == 0:
            raise ValueError("CSV文件中没有标记为TRUE的选址点")

        self.points = selected_data[[self.x_col, self.y_col]].values
        self.M = len(self.points)

        # 3. 设置边界点（使用所有点）
        if not (self.use_shp_boundary and self.boundary_shp):
            self.boundary_points_raw = self.data[[self.x_col, self.y_col]].values

        print(f"数据处理完成！")
        print(f"总点数: {len(self.data)}")
        print(f"选址点数: {self.M}")
        print(f"选址点占比: {self.M / len(self.data) * 100:.1f}%")
        print(f"X列: {self.x_col}, Y列: {self.y_col}")
        if self.points is not None and len(self.points) > 0:
            print(f"选址点X范围: [{self.points[:, 0].min():.4f}, {self.points[:, 0].max():.4f}]")
            print(f"选址点Y范围: [{self.points[:, 1].min():.4f}, {self.points[:, 1].max():.4f}]")

        return True

    def extract_boundary_from_shp(self):
        """
        从shp文件中提取边界
        """
        if self.boundary_shp is None or not os.path.exists(self.boundary_shp):
            raise ValueError(f"shp文件不存在: {self.boundary_shp}")

        print(f"\n从shp文件提取边界: {self.boundary_shp}")

        # 读取shp文件
        gdf = gpd.read_file(self.boundary_shp)
        print(f"找到 {len(gdf)} 个要素")
        print(f"坐标系: {gdf.crs}")

        # 合并所有几何要素
        if len(gdf) == 1:
            self.boundary_geometry = gdf.geometry.iloc[0]
        else:
            # 合并多个多边形
            self.boundary_geometry = unary_union(gdf.geometry)

        self.boundary_crs = gdf.crs

        # 确保几何对象有效
        if self.boundary_geometry is None or self.boundary_geometry.is_empty:
            raise ValueError("从shp文件提取的几何对象为空")

        # 处理多面要素
        if isinstance(self.boundary_geometry, MultiPolygon):
            print(f"多面要素处理: 包含 {len(self.boundary_geometry.geoms)} 个独立区域")
            # 将所有多边形合并为一个（对于蒙特卡洛模拟更方便）
            self.boundary_polygon = self.boundary_geometry
            # 获取边界点（用于可视化）
            all_points = []
            for poly in self.boundary_geometry.geoms:
                if hasattr(poly, 'exterior'):
                    points = np.array(poly.exterior.coords)
                    all_points.extend(points)
            self.boundary_points_final = np.array(all_points)
        else:
            # 单个多边形
            self.boundary_polygon = self.boundary_geometry
            if hasattr(self.boundary_geometry, 'exterior'):
                self.boundary_points_final = np.array(self.boundary_geometry.exterior.coords)

        # 更新面积
        if isinstance(self.boundary_polygon, MultiPolygon):
            self.area = sum(poly.area for poly in self.boundary_polygon.geoms)
        else:
            self.area = self.boundary_polygon.area

        # 检查选址点是否都在边界内
        if self.points is not None and self.M > 0:
            points_inside = 0
            for point in self.points:
                if self.boundary_polygon.contains(Point(point)):
                    points_inside += 1

            if points_inside < self.M:
                print(f"警告: {self.M - points_inside}个选址点位于边界外")
                print(f"边界内选址点占比: {points_inside / self.M * 100:.1f}%")

        print(f"shp边界提取完成:")
        print(f"  几何类型: {type(self.boundary_polygon).__name__}")
        print(f"  总面积: {self.area:.4f}")
        if isinstance(self.boundary_polygon, Polygon):
            print(f"  多边形周长: {self.boundary_polygon.length:.4f}")
        elif isinstance(self.boundary_polygon, MultiPolygon):
            total_perimeter = sum(poly.length for poly in self.boundary_polygon.geoms)
            print(f"  总周长: {total_perimeter:.4f}")

        return self.boundary_polygon


    def calculate_Q_value(self, points=None, area=None, theta_factor=3):
        """
        计算点集的Q值（基于势能的均匀性度量）
        """
        if points is None:
            points = self.points

        if points is None or len(points) == 0:
            raise ValueError("未提供点集数据或点集为空")

        M = len(points)

        if area is None:
            area = self.area if self.area else 1.0

        # 特征距离
        r_c = np.sqrt(area / M)

        # 势能参数Θ
        theta = theta_factor * r_c ** 2

        # 构建KDTree进行最近邻搜索
        kdtree = KDTree(points)

        # 查询k+1个最近邻（包含自身）
        distances, indices = kdtree.query(points, k=self.k + 1)
        distances = distances[:, 1:]  # 排除自身

        # 计算Q值
        total_Q = 0
        count = 0

        for i in range(M):
            for dist in distances[i]:
                # 势能函数
                PE = theta / (theta + dist ** 2)
                # 质量度量
                Q_ij = 1 - PE
                total_Q += Q_ij
                count += 1

        Q = total_Q / count if count > 0 else 0

        return Q, r_c, theta

    def generate_random_points_in_polygon(self, M, method='rejection'):
        """
        在多边形或多面内生成均匀随机点
        """
        if self.boundary_polygon is None:
            raise ValueError("未定义边界多边形")

        # 处理多面要素
        if isinstance(self.boundary_polygon, MultiPolygon):
            return self._generate_random_points_in_multipolygon(M, method)

        # 单个多边形的情况
        minx, miny, maxx, maxy = self.boundary_polygon.bounds
        points = []
        attempts = 0
        max_attempts = M * 100

        if method == 'rejection':
            # 拒绝采样法
            while len(points) < M and attempts < max_attempts:
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)

                if self.boundary_polygon.contains(point):
                    points.append([x, y])

                attempts += 1

        elif method == 'grid_jitter':
            # 网格抖动法
            spacing = np.sqrt(self.area / M)

            x_grid = np.arange(minx, maxx, spacing)
            y_grid = np.arange(miny, maxy, spacing)

            xx, yy = np.meshgrid(x_grid, y_grid)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])

            for point in grid_points[:M * 2]:
                if len(points) >= M:
                    break

                jitter_x = np.random.uniform(-spacing / 2, spacing / 2)
                jitter_y = np.random.uniform(-spacing / 2, spacing / 2)

                x_jittered = point[0] + jitter_x
                y_jittered = point[1] + jitter_y

                if self.boundary_polygon.contains(Point(x_jittered, y_jittered)):
                    points.append([x_jittered, y_jittered])

        # 如果生成的点不够，用拒绝采样法补充
        if len(points) < M:
            remaining = M - len(points)
            for _ in range(remaining * 10):
                if len(points) >= M:
                    break
                x = np.random.uniform(minx, maxx)
                y = np.random.uniform(miny, maxy)
                point = Point(x, y)
                if self.boundary_polygon.contains(point):
                    points.append([x, y])

        return np.array(points)[:M]

    def _generate_random_points_in_multipolygon(self, M, method='rejection'):
        """
        在多面要素内生成均匀随机点
        """
        polygons = list(self.boundary_polygon.geoms)

        # 按面积分配点数
        areas = [poly.area for poly in polygons]
        total_area = sum(areas)

        # 计算每个多边形应分配的点数
        points_per_poly = [int(M * area / total_area) for area in areas]

        # 处理舍入误差
        diff = M - sum(points_per_poly)
        if diff > 0:
            max_idx = np.argmax(areas)
            points_per_poly[max_idx] += diff

        all_points = []

        # 为每个多边形生成点
        for i, poly in enumerate(polygons):
            if points_per_poly[i] > 0:
                temp_polygon = poly
                minx, miny, maxx, maxy = temp_polygon.bounds

                sub_points = []
                attempts = 0
                max_attempts = points_per_poly[i] * 100

                while len(sub_points) < points_per_poly[i] and attempts < max_attempts:
                    x = np.random.uniform(minx, maxx)
                    y = np.random.uniform(miny, maxy)
                    point = Point(x, y)

                    if temp_polygon.contains(point):
                        sub_points.append([x, y])

                    attempts += 1

                all_points.extend(sub_points[:points_per_poly[i]])

        return np.array(all_points)[:M]

    def run_monte_carlo_benchmark(self, num_simulations=1000, verbose=True):
        """
        运行蒙特卡洛模拟，建立Q值基准分布
        """
        if self.points is None or self.M == 0:
            raise ValueError("请先加载点集数据")

        M = self.M

        if verbose:
            print(f"开始蒙特卡洛基准模拟...")
            print(f"模拟次数: {num_simulations}")
            print(f"每模拟点数: {M}")
            print(f"区域面积: {self.area:.4f}")
            if self.use_shp_boundary and self.boundary_shp:
                print(f"边界来源: shp文件")
            else:
                print(f"边界方法: {self.boundary_method}")
            print(f"k值: {self.k}")
            print("-" * 60)

        Q_values = []

        for i in range(num_simulations):
            # 生成随机点集
            random_points = self.generate_random_points_in_polygon(M, method='rejection')

            # 计算Q值
            Q, _, _ = self.calculate_Q_value(random_points, self.area)
            Q_values.append(Q)

            # 显示进度
            if verbose and (i + 1) % max(1, num_simulations // 10) == 0:
                print(f"进度: {i + 1}/{num_simulations} ({(i + 1) / num_simulations * 100:.0f}%)")

        Q_values = np.array(Q_values)

        # 计算统计信息
        self.benchmark_mean = np.mean(Q_values)
        self.benchmark_std = np.std(Q_values)

        # 计算百分位数
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.benchmark_percentiles = {
            p: np.percentile(Q_values, p) for p in percentiles
        }

        self.Q_distribution = Q_values
        self.num_simulations = num_simulations

        if verbose:
            print(f"\n基准模拟完成!")
            print(f"Q值基准分布统计:")
            print(f"  均值: {self.benchmark_mean:.6f}")
            print(f"  标准差: {self.benchmark_std:.6f}")
            print(f"  最小值: {np.min(Q_values):.6f}")
            print(f"  最大值: {np.max(Q_values):.6f}")
            print(f"  中位数 (P50): {self.benchmark_percentiles[50]:.6f}")
            print(f"  5-95百分位范围: [{self.benchmark_percentiles[5]:.6f}, {self.benchmark_percentiles[95]:.6f}]")

        return {
            'mean': self.benchmark_mean,
            'std': self.benchmark_std,
            'percentiles': self.benchmark_percentiles,
            'distribution': Q_values,
            'num_simulations': num_simulations
        }

    def evaluate_uniformity(self, verbose=True):
        """
        评估点集均匀性（相对于蒙特卡洛基准）
        """
        if self.points is None or self.M == 0:
            raise ValueError("请先加载点集数据")

        if self.benchmark_mean is None:
            print("警告: 未建立蒙特卡洛基准，将自动运行基准模拟...")
            self.run_monte_carlo_benchmark(num_simulations=500, verbose=verbose)

        # 计算点集的Q值
        Q_actual, r_c, theta = self.calculate_Q_value(self.points, self.area)

        # 计算百分位数
        percentile = np.sum(self.Q_distribution <= Q_actual) / len(self.Q_distribution) * 100

        # 计算Z分数
        Z_score = (Q_actual - self.benchmark_mean) / self.benchmark_std if self.benchmark_std > 0 else 0

        # 计算差异值
        diff = Q_actual - self.benchmark_mean
        diff_percent = (diff / self.benchmark_mean) * 100 if self.benchmark_mean != 0 else 0

        # 定性评估
        if percentile >= 95:
            rating = "A+ (极好)"
            evaluation = "比95%的随机分布更均匀"
        elif percentile >= 85:
            rating = "A (优秀)"
            evaluation = "比85%的随机分布更均匀"
        elif percentile >= 70:
            rating = "B (良好)"
            evaluation = "比70%的随机分布更均匀"
        elif percentile >= 50:
            rating = "C (一般)"
            evaluation = "与随机分布相当"
        elif percentile >= 30:
            rating = "D (稍差)"
            evaluation = "比多数随机分布差"
        else:
            rating = "F (较差)"
            evaluation = "有明显的聚类或不均匀"

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"点集均匀性评估报告")
            print(f"{'=' * 60}")
            print(f"基本信息:")
            print(f"  总点数: {len(self.data)}")
            print(f"  选址点数: {self.M}")
            print(f"  选址点占比: {self.M / len(self.data) * 100:.1f}%")
            if self.use_shp_boundary and self.boundary_shp:
                print(f"  边界来源: shp文件")
                if isinstance(self.boundary_polygon, MultiPolygon):
                    print(f"  多边形数量: {len(self.boundary_polygon.geoms)}")
            else:
                print(f"  边界点数: {len(self.boundary_points_raw)}")
                print(f"  边界方法: {self.boundary_method}")
            print(f"  区域面积: {self.area:.4f}")
            print(f"  特征距离 r_c: {r_c:.4f}")
            print(f"  势能参数 Θ: {theta:.4f}")
            print(f"  实际Q值: {Q_actual:.6f}")

            print(f"\n基准信息 (基于{self.num_simulations}次蒙特卡洛模拟):")
            print(f"  基准Q均值: {self.benchmark_mean:.6f}")
            print(f"  基准Q标准差: {self.benchmark_std:.6f}")
            print(f"  基准中位数 (P50): {self.benchmark_percentiles[50]:.6f}")

            print(f"\n评估结果:")
            print(f"  百分位数: {percentile:.2f}%")
            print(f"  Z分数: {Z_score:.3f}")
            print(f"  评级: {rating}")
            print(f"  评价: {evaluation}")

            print(f"\n与基准比较:")
            print(f"  与基准均值相差: {diff:.6f} ({diff_percent:+.1f}%)")

            if diff > 0:
                print(f"  ✅ 点集比随机分布更均匀")
            else:
                print(f"  ⚠ 点集不如随机分布均匀")

            # 关键百分位数比较
            print(f"\n关键百分位数比较:")
            print(f"  你的Q值: {Q_actual:.6f}")
            print(f"  P5 (较差): {self.benchmark_percentiles[5]:.6f}")
            print(f"  P25 (中下): {self.benchmark_percentiles[25]:.6f}")
            print(f"  P50 (中位数): {self.benchmark_percentiles[50]:.6f}")
            print(f"  P75 (中上): {self.benchmark_percentiles[75]:.6f}")
            print(f"  P95 (优秀): {self.benchmark_percentiles[95]:.6f}")

        # 准备返回结果
        results = {
            'total_points': len(self.data),
            'selected_points': self.M,
            'selection_percentage': self.M / len(self.data) * 100,
            'area': self.area,
            'boundary_source': 'shp' if (self.use_shp_boundary and self.boundary_shp) else 'points',
            'boundary_method': self.boundary_method,
            'x_col': self.x_col,
            'y_col': self.y_col,
            'Q_actual': Q_actual,
            'r_c': r_c,
            'theta': theta,
            'benchmark_mean': self.benchmark_mean,
            'benchmark_std': self.benchmark_std,
            'percentile': percentile,
            'Z_score': Z_score,
            'rating': rating,
            'evaluation': evaluation,
            'diff_from_mean': diff,
            'diff_percent': diff_percent
        }

        return results

    def visualize_evaluation(self, results=None):
        """
        可视化评估结果 - 只显示两个核心图表，不显示文本总结
        """
        if self.Q_distribution is None:
            raise ValueError("请先运行蒙特卡洛基准模拟")

        if results is None:
            results = self.evaluate_uniformity(verbose=False)

        Q_actual = results['Q_actual']
        percentile = results['percentile']

        # 创建1x2的子图布局
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 1. 点集分布和边界（左）
        ax1 = axes[0]

        # 绘制所有点（浅色背景）
        if self.data is not None:
            all_points = self.data[[self.x_col, self.y_col]].values
            ax1.scatter(all_points[:, 0], all_points[:, 1],
                        s=20, c='lightgray', alpha=0.3,
                        label=f'所有点 ({len(self.data)})', zorder=1)

        # 绘制选址点
        if self.points is not None and len(self.points) > 0:
            ax1.scatter(self.points[:, 0], self.points[:, 1],
                        s=60, c='blue', alpha=0.7, edgecolors='black',
                        label=f'选址点 ({self.M})', zorder=3)

        # 绘制边界（shp边界或点集边界）
        if self.use_shp_boundary and self.boundary_shp and self.boundary_polygon:
            # 绘制shp边界
            if isinstance(self.boundary_polygon, MultiPolygon):
                # 绘制多个多边形
                for i, poly in enumerate(self.boundary_polygon.geoms):
                    if hasattr(poly, 'exterior'):
                        hull_x, hull_y = poly.exterior.xy
                        ax1.plot(hull_x, hull_y, 'g-', linewidth=3, alpha=0.7,
                                 label=f'区域{i + 1}' if i == 0 else "", zorder=4)
                        ax1.fill(hull_x, hull_y, 'lightgreen', alpha=0.2, zorder=2)
                boundary_label = f'shp边界 ({len(self.boundary_polygon.geoms)}个区域)'
            else:
                # 绘制单个多边形
                if hasattr(self.boundary_polygon, 'exterior'):
                    hull_x, hull_y = self.boundary_polygon.exterior.xy
                    ax1.plot(hull_x, hull_y, 'g-', linewidth=3, alpha=0.7,
                             label='shp边界', zorder=4)
                    ax1.fill(hull_x, hull_y, 'lightgreen', alpha=0.2, zorder=2)
                boundary_label = 'shp边界'
        elif self.boundary_points_final is not None:
            # 绘制点集边界
            ax1.scatter(self.boundary_points_final[:, 0], self.boundary_points_final[:, 1],
                        s=120, c='red', alpha=0.8, edgecolors='black',
                        marker='o', label=f'边界点 ({len(self.boundary_points_final)})', zorder=5)

            if self.boundary_polygon and hasattr(self.boundary_polygon, 'exterior'):
                hull_x, hull_y = self.boundary_polygon.exterior.xy
                ax1.plot(hull_x, hull_y, 'g-', linewidth=3, alpha=0.7,
                         label=f'{self.boundary_method.capitalize()}边界', zorder=4)
                ax1.fill(hull_x, hull_y, 'lightgreen', alpha=0.2, zorder=2)
                boundary_label = f'{self.boundary_method.capitalize()}边界'

        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 在图表标题中显示关键信息
        if self.use_shp_boundary and self.boundary_shp:
            if isinstance(self.boundary_polygon, MultiPolygon):
                boundary_info = f'shp边界 (区域数: {len(self.boundary_polygon.geoms)})'
            else:
                boundary_info = 'shp边界'
        else:
            boundary_info = f'{self.boundary_method.capitalize()}边界'

        title = f'点集分布与边界\n总点数: {len(self.data)}, 选址点: {self.M} ({results["selection_percentage"]:.1f}%)\n边界: {boundary_info}, Q值: {Q_actual:.6f}'
        ax1.set_title(title, fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')

        # 2. Q值基准分布直方图（右）
        ax2 = axes[1]
        n, bins, patches = ax2.hist(self.Q_distribution, bins=30,
                                    alpha=0.7, color='skyblue', edgecolor='black')

        # 标记基准均值
        ax2.axvline(self.benchmark_mean, color='green', linestyle='--',
                    linewidth=2, label=f'基准均值: {self.benchmark_mean:.4f}')

        # 标记实际Q值
        ax2.axvline(Q_actual, color='red', linewidth=3,
                    label=f'实际Q值: {Q_actual:.4f}')

        # 标记关键百分位数
        for p, color in [(5, 'red'), (25, 'orange'), (50, 'yellow'),
                         (75, 'orange'), (95, 'red')]:
            value = self.benchmark_percentiles[p]
            ax2.axvline(value, color=color, linestyle=':', alpha=0.7,
                        linewidth=1.5, label=f'P{p}: {value:.4f}')

        # 填充实际值所在区间
        bin_index = np.digitize(Q_actual, bins) - 1
        if 0 <= bin_index < len(patches):
            patches[bin_index].set_facecolor('red')
            patches[bin_index].set_alpha(0.8)

        ax2.set_xlabel('Q值')
        ax2.set_ylabel('频数')

        # 在图表标题中显示评估结果
        title = f'Q值基准分布 (模拟{self.num_simulations}次)\n百分位数: {percentile:.1f}%, 评级: {results["rating"]}'
        ax2.set_title(title, fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 设置总标题
        boundary_source = 'shp文件' if (self.use_shp_boundary and self.boundary_shp) else '点集提取'
        plt.suptitle(
            f'点集均匀性综合评估报告 - 数据来源: {os.path.basename(self.data_csv)} (边界来源: {boundary_source})',
            fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        return fig

    def export_data(self, output_dir=None):
        """
        导出处理后的数据
        """
        import os

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 导出数据文件名称
        base_name = os.path.splitext(os.path.basename(self.data_csv))[0]

        # 导出完整的处理结果
        output_path = os.path.join(output_dir, f'{base_name}_uniformity_results.csv')

        # 准备结果数据
        result_df = pd.DataFrame({
            'point_id': range(len(self.data)),
            'x': self.data[self.x_col],
            'y': self.data[self.y_col],
            'selected': self.data['selected'],
            'is_boundary_point': False  # 可以添加边界点标记
        })

        # 保存结果
        result_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\n数据已导出:")
        print(f"评估结果文件: {output_path}")
        print(f"   总点数: {len(result_df)}")
        print(f"   选址点数: {result_df['selected'].sum()}")
        print(f"   选址点占比: {result_df['selected'].sum() / len(result_df) * 100:.1f}%")
        print()

        # 显示数据预览
        print("数据预览（前5行）：")
        print(result_df.head())
        print()

        return output_path


def evaluate_point_set_uniformity(data_csv, boundary_shp=None,
                                  x_col='x', y_col='y',
                                  k_neighbors=4, boundary_method='concave',
                                  num_simulations=1000, export_data=False,
                                  use_shp_boundary=True):
    """
    主函数：评估点集均匀性

    参数:
        data_csv: 数据CSV文件路径（包含x, y, selected字段）
        boundary_shp: shp边界文件路径（可选）
        x_col: 要使用的x坐标列名
        y_col: 要使用的y坐标列名
        k_neighbors: 最近邻数
        boundary_method: 使用shp边界
        num_simulations: 蒙特卡洛模拟次数
        export_data: 是否导出处理后的数据
        use_shp_boundary: 是否使用shp文件作为边界（如果提供了boundary_shp）
    """
    print("=" * 80)
    print("点集均匀性评估系统")
    print("=" * 80)
    print(f"数据文件: {data_csv}")
    if boundary_shp and use_shp_boundary:
        print(f"边界文件: {boundary_shp}")
    else:
        print(f"边界提取方法: {boundary_method}")
    print(f"使用坐标列: {x_col}, {y_col}")
    print(f"最近邻数(k): {k_neighbors}")
    print("-" * 80)

    # 1. 初始化评估器
    evaluator = PointSetUniformityEvaluator(
        data_csv=data_csv,
        boundary_shp=boundary_shp,
        k_neighbors=k_neighbors,
        boundary_method=boundary_method,
        x_col=x_col,
        y_col=y_col,
        use_shp_boundary=use_shp_boundary
    )

    # 2. 加载和处理数据
    evaluator.load_and_process_data()

    # 3. 提取边界
    if boundary_shp and use_shp_boundary:
        evaluator.extract_boundary_from_shp()
    else:
        evaluator.extract_boundary_from_points()

    # 4. 运行蒙特卡洛基准模拟
    print("\n" + "=" * 60)
    print("建立基准分布...")
    M = evaluator.M
    num_simulations = min(num_simulations, max(500, M * 10))
    evaluator.run_monte_carlo_benchmark(
        num_simulations=num_simulations,
        verbose=True
    )

    # 5. 评估均匀性
    print("\n" + "=" * 60)
    print("评估均匀性...")
    results = evaluator.evaluate_uniformity(verbose=True)

    # 6. 导出数据（可选）
    if export_data:
        evaluator.export_data()

    # 7. 生成可视化报告
    print("\n" + "=" * 60)
    print("生成可视化报告...")
    evaluator.visualize_evaluation(results)

    print("\n" + "=" * 80)
    print("评估完成!")
    print("=" * 80)

    return evaluator, results
if __name__=='__main__':
    k = 200
    locfile = r'E:\流形研究\20260204选址流形整理\loc\greedy_nodup_locs_k{k}.npy'.format(k=k)
    res_csv = r'E:\流形研究\20260204选址流形整理\emb\R0_bnc5_inf100_k30_nb1000_tc0.9\nc2_perp6_thre15_w00.3_niter50_kiter10_N4/res.csv'
    manif = r'E:\流形研究\20260204选址流形整理\alphashape\rev_mask_thre15.csv'
    manifp = r'E:\流形研究\20260204选址流形整理\alphashape\rev_mask_thre15.png'
    output_dir=r'E:\流形研究\20260204选址流形整理\alphashape\rev_mask_thre15'

    #计算均匀性指标
    shp_file = r'{path}\manif.shp'.format(path=output_dir)
    # 评估点集均匀性
    evaluator, results = evaluate_point_set_uniformity(
        data_csv=manif,
        boundary_shp=shp_file,
        x_col='x',  # CSV文件中的x列名
        y_col='y',  # CSV文件中的y列名
        use_shp_boundary=True if shp_file else False,
        num_simulations=1000,
        export_data=True
    )


