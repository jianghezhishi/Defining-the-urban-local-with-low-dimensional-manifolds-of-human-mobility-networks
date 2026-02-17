'''
选址流形代码整理
'''
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps, cm
import datetime
import os
import networkx as nx
from hm_as_manifold_v4 import boundary,mat_to_g,calculate_path,output_path,read_path,\
    smacof_by_RRE,plot_s_curve,output_res,read_manifold,topo_fit,pearson_plot_part,auto_dich
csv.field_size_limit(1024*1024*100)
def xy_to_cid(x,y,scale,ynumls=40):
    xid=int(x/scale)
    yid=int(y/scale)
    return xid*ynumls+yid+1
def cid_to_xy(cid,ynum=40):
    xid=int((cid-1)/ynum)
    yid=cid-1-xid*ynum
    return xid,yid
def traj_to_cover(scale=5,ynumls=40,output=True,day=None):
    cid_cover={}#记录大尺度格子的覆盖个体
    with open(r'E:\基础数据\轨迹\日本\yjmob100k-dataset1.csv','r') as f:
        rd=csv.reader(f)
        header=next(rd)
        for row in rd:
            if day==None or int(row[1])==day:
                uid=row[0]
                x,y=int(row[3])-1,int(row[4])-1
                cid=xy_to_cid(x,y,scale,ynumls)
                if cid not in cid_cover:
                    cid_cover[cid]=set()
                cid_cover[cid].add(uid)
    if output==True:
        with open(r'E:\基础数据\轨迹\日本\cover_day{day}.csv'.format(day=day),'w',newline='') as f:
            wt=csv.writer(f)
            for cid in cid_cover:
                wt.writerow([cid,cid_cover[cid]])
    return cid_cover
def read_cover(file):
    cover = {}
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            cover[int(row[0])] = eval(row[1])
    print('subcover read', datetime.datetime.now())
    return cover
def solve_greed_nodup(cover, k, title):
    # k：选址点数；cover：覆盖关系字典；title：保存选址点文件名
    # solving the location choice problem
    cover0 = cover.copy()
    loc = []
    nums = []
    for i in range(k):
        temp = {}
        for cid in cover:
            temp[cid] = len(cover[cid])
        stemp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        cid0 = stemp[0][0]
        loc.append(cid0)
        nums.append(stemp[0][1])
        covered = cover[cid0]
        cover1 = {}
        for cid in cover:
            if cid != cid0:
                cover1[cid] = cover[cid] - covered
        cover = cover1
    temp = set()
    for cid in loc:
        temp |= cover0[cid]
    print(len(loc),len(temp))
    np.save(title, np.array(loc))
    selection_df = pd.DataFrame(loc, columns=['node'])
    selection_df['selected'] = True
    selection_df.to_csv(title.replace('.npy', '.csv'), index=False)
    return loc, nums

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

def log_plus_mat_perp(file,title,minlam=5,inf=100,perp=6,opt='log',nc=2):
    #不使用回归，d=log(nodupsi)+log(nodupsj)+2*log(wij)
    cid_cover = {}  # 即格子覆盖人数，用minpop筛选
    with open(file, 'r') as f:
        rd = csv.reader(f)
        for row in rd:
            if len(eval(row[1])) >= 1:
                cid_cover[int(row[0])] = eval(row[1])
    print('cover read')
    nodes = list(cid_cover.keys())

    cp_lam = {}

    for cid in nodes:
        temp={}
        for cid1 in nodes:
            if cid != cid1:
                d = len(cid_cover[cid]&cid_cover[cid1])
                if d>=minlam:
                    temp[cid1] = d
        if temp != {}:
            cands = []
            stemp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
            for i in range(min(perp - 1, len(stemp))):
                cp_lam[(cid,stemp[i][0])]=stemp[i][1]

    print('lam ready')
    nodes = find_largest_connected_component(list(cp_lam.keys()))

    cid_nodup_cover = {}
    for cid in nodes:
        cover0 = cid_cover[cid].copy()
        for cid1 in nodes:
            if cid1 != cid:
                cover0 -= cid_cover[cid1]
        cid_nodup_cover[cid] = cover0
    print('no dup ready')

    cp_lam1 = {}
    for cp in cp_lam:
        cid, cid1 = cp
        if cid in nodes and cid1 in nodes:
            if opt=='log':
                cp_lam1[cp] = np.log(max(1,len(cid_nodup_cover[cid])))+\
                                       np.log(max(1,len(cid_nodup_cover[cid1])))+\
                                       2*np.log(cp_lam[cp])
                cp_lam1[(cid1,cid)] = np.log(max(1, len(cid_nodup_cover[cid]))) + \
                              np.log(max(1, len(cid_nodup_cover[cid1]))) + \
                              2 * np.log(cp_lam[cp])
            elif opt=='sqrt':
                cp_lam1[cp] = (len(cid_nodup_cover[cid]))**(1/nc) + \
                              (len(cid_nodup_cover[cid1]))**(1/nc)+ \
                              2 * (cp_lam[cp])**(1/nc)
                cp_lam1[(cid1, cid)] = (len(cid_nodup_cover[cid]))**(1/nc) + \
                              (len(cid_nodup_cover[cid1]))**(1/nc)+ \
                              2 * (cp_lam[cp])**(1/nc)
    cp_lam = cp_lam1

    cid_s = {}

    for cid in nodes:
        cid_s[cid] = len(cid_cover[cid])
    print('s ready', len(cp_lam), len(cid_s))
    with open(title,'w',newline='') as f:
        wt=csv.writer(f)
        wt.writerow([nodes])
        for cid in nodes:
            row = []
            for cid1 in nodes:
                if cid==cid1:
                    row.append(0)
                elif (cid,cid1) in cp_lam:
                    row.append(cp_lam[(cid,cid1)])
                elif (cid1,cid) in cp_lam:
                    row.append(cp_lam[(cid1,cid)])
                else:
                    row.append(inf)
            xy=tuple(cid_to_xy(cid))
            wt.writerow([row,xy])
    return

def read_mat(file, maxd, sigma=6, dmax=4):
    mask = []
    mat = []
    with open(file, 'r') as f:
        rd = csv.reader(f)
        header = next(rd)
        nodes = eval(header[0])
        count = 0
        for row in rd:
            cid0 = nodes[count]
            temp = eval(row[0])
            mat.append(temp)
            if sigma != None:
                dict0 = {}
                for i in range(len(temp)):
                    if temp[i]<maxd:
                        dict0[i] = temp[i]
                stemp = sorted(dict0.items(), key=lambda x: x[1],reverse=True)

                for i in range(min(len(stemp),sigma-1)):
                    if 0 < stemp[i][1] < maxd:
                        cid1 = nodes[stemp[i][0]]
                        mask.append((cid0, cid1))
            else:
                for i in range(len(temp)):
                    if 0 < temp[i] < dmax:
                        cid1 = nodes[i]
                        mask.append((cid0, cid1))
            count += 1
    return nodes, mat, mask
def get_cons_geod(paths, lengths, bound, inf, thre, w0, title, mask1):
    # 获取consistent的测地线，满足两个条件之一，不与边界相交，或两端到边界距离大于测地线长度
    bound1 = set(bound)
    cid_b_dist = {}
    for cid in paths:
        temp = []
        for b in bound:
            if b in lengths[cid]:
                temp.append(lengths[cid][b])
        temp.sort()
        if len(temp) == 0:
            cid_b_dist[cid] = inf
        else:
            cid_b_dist[cid] = temp[0]
    print('dist to boundary ready')

    cps = np.zeros((len(paths), len(paths)))

    for i in range(len(paths)):
        if i % 100 == 0:
            print(i)
        for j in range(len(paths)):
            if len(set(paths[i].get(j, bound1)) & bound1) == 0:
                cps[i][j] = 1
            elif (i, j) in mask1 or (j, i) in mask1:
                cps[i][j] = 1
            else:
                if lengths[i].get(j, inf * 3) <= cid_b_dist[i] + cid_b_dist[j]:
                    cps[i][j] = 1
                elif thre<=lengths[i].get(j, inf * 3) <inf:
                    cps[i][j] = w0
    print('ready')
    np.save(title, cps)
    print('saved')
    return cps
def tcie_para(R, perp, perpsne, bnc, nc, inf, k, nb, tc, thre, w0, niter, kiter, N, perptest=6, \
              show=False, matfile=None, dir0=r'E:\流形研究\20250305tcie_mask',emb=True):
    # R，计算距离矩阵时的参数
    # perp决定mask大小
    # perpsne，用于画边界图的tsne嵌入，使用的perp参数
    # bnc，边缘点探测所用维数
    # nc，维数
    # inf，距离矩阵中无连接部分的赋值
    # k，识别边界时的局部大小
    # nb，边界点数量
    # tc，基于方向的边界识别中的比例阈值
    # thre，计算权重矩阵时，路径小于该阈值的被赋予较低权重
    # w0，权重矩阵中的较低权重
    # niter，每轮rre进行的总smacof轮数
    # kiter，rre用于插值的轮数
    # N，rre轮数
    # perptest，用于拓扑和距离嵌入分析的perp
    # 调参主要针对niter之前的部分，即边界识别的部分，因为后面基本都是让smacof收敛，只要收敛就行，影响不大
    dirs = r'{dir0}\R{R}_bnc{bnc}_inf{inf}_k{k}_nb{nb}_tc{tc}'. \
        format(dir0=dir0, R=R, bnc=bnc, inf=inf, k=k, nb=nb, tc=tc)
    embdir = r'{dir}\nc{nc}_perp{perp}_thre{thre}_w0{w0}_niter{niter}_kiter{kiter}_N{N}'. \
        format(dir=dirs, nc=nc, perp=perp, thre=thre, w0=w0, niter=niter, kiter=kiter, N=N)
    embtitle = r'{dir}\res.csv'.format(dir=embdir)
    if matfile == None:
        matfile = r'E:\流形研究\20250101基于加权双曲的距离\triphi_R{R}.csv'.format(R=R)
    nodes, mat, mask = read_mat(matfile, maxd=inf, sigma=perp, dmax=None)
    #print(mask)
    #nodes, mat, mask = read_mat(matfile, maxd=10+R, sigma=perp, dmax=None)
    # #20250617这是旧版，似乎不太合理，所以改为maxd=inf
    mat = np.array(mat)
    print('mat read')
    if emb:

        if not os.path.exists(dirs):
            os.makedirs(dirs)
        print('dirs created')
        boundname = r'{dir}\bound.npy'.format(dir=dirs)
        if not os.path.exists(boundname):
            print('making bounds')
            bound = list(boundary(mat, k, nb, tc, inf, bnc))
            bound = np.array(bound)
            print('bound ready')
            # print(bound)

            np.save(boundname, bound)
            print('bound saved')



        else:
            bound = np.load(boundname)
            print('bound read')
        print('bound ready')

        pathname = r'{dir}\perp{perp}_paths.csv'.format(dir=dirs, perp=perp)
        lengthname = r'{dir}\perp{perp}_lengths.csv'.format(dir=dirs, perp=perp)
        if not os.path.exists(pathname):
            print('making path and length')
            g, mask1 = mat_to_g(mat, mask, nodes, inf)
            paths, lengths = calculate_path(g)
            print('path and length ready')
            output_path(paths, lengths, pathname, lengthname)
        else:
            paths, lengths = read_path(pathname, lengthname)

            cid_tag = dict(zip(nodes, range(len(nodes))))
            mask1 = []
            for cp in mask:
                c1, c2 = cp
                mask1.append((cid_tag[c1], cid_tag[c2]))
            print('bound, paths, lengths read')

        wname = r'{dir}\wmat_perp{perp}_thre{thre}_w0{w0}.npy'. \
            format(dir=dirs, perp=perp, thre=thre, w0=w0)
        if not os.path.exists(wname):
            print('making wmat')
            wmat = get_cons_geod(paths, lengths, bound, inf, thre, w0, wname, mask1)

        else:
            print('reading wmat')
            wmat = np.load(wname)

        print('cons geod ready')

        if not os.path.exists(embdir):
            os.makedirs(embdir)
            print('embedding')
            x, slist = smacof_by_RRE(niter, kiter, N, lengths, wmat, nc, inf, embdir, init=True)
        else:
            print('reading embedding')
            slist = np.load(r'{dir}\slist.npy'.format(dir=embdir))
            x = np.load(r'{dir}\x_N{N}.npy'.format(dir=embdir, N=N - 1))
        print('emb ready')
        plot_s_curve(r'{dir}\slist.npy'.format(dir=embdir), r'{dir}\slist.png'.format(dir=embdir), show)


        output_res(x, nodes, embtitle)

    res, nodes = read_manifold(embtitle, nc=nc, opt=1)

    if perp != perptest:
        nodes, mat, mask = read_mat(matfile, maxd=10 + R, sigma=perptest, dmax=None)

    curvetitle = r'{dir}\curve_perptest{perptest}.png'.format(dir=embdir, perptest=perptest)
    xinter,yinter=topo_fit(nodes, mask, res, r'{dir}\hist_perptest{perptest}.png'.format(dir=embdir, perptest=perptest), \
             curvetitle, part=None, win=30, show=show,temptitle=r'{dir}\topo.npy'.format(dir=embdir))


    pearstitle = r'{dir}\pears_perptest{perptest}.png'.format(dir=embdir, perptest=perptest)
    r = pearson_plot_part(res, nodes, nodes, mask, mat, pearstitle, opt=1, maxd=R + 20, show=show)

    print(r, R, perp, perpsne, nc, inf, k, nb, tc, thre, w0, niter, kiter, N)
    title = r'{dir}\para.csv'.format(dir=embdir)
    with open(title, 'w', newline='') as f:
        wt = csv.writer(f)
        row1 = [r, R, perp, perpsne, nc, inf, k, nb, tc, thre, w0, niter, kiter, N]
        row0 = ['pears'] + 'R,perp,perpsne,nc,inf,k,nb,tc,thre,w0,niter,kiter,N'.split(',')
        arr = np.array([row0, row1]).T.tolist()
        for i in arr:
            wt.writerow([i])
    return r,yinter

def pick_selection_points(file, embedding):
    selection = np.load(file)
    embedding['selected'] = False
    embedding.loc[embedding.cid.isin(selection), 'selected'] = True
    return embedding
def draw_latlon_fold(embedding, select_longitudes, select_latitudes,
                     title=r"E:\流形研究\20260121名古屋选址流形\figure\latlon_manifold.png"):
    print(embedding.head)
    x_min = embedding["x"].min()
    x_max = embedding["x"].max()

    y_min = embedding["y"].min()
    y_max = embedding["y"].max()
    fx=15
    fy=(y_max-y_min)/(x_max-x_min)*fx
    fig, ax = plt.subplots(1,2,figsize=(fx*3,fy))
    embedding_selected = embedding.loc[embedding['selected'] == True]
    embedding_not_selected = embedding.loc[embedding['selected'] == False]
    cmap1 = colormaps.get_cmap("spring")
    sns.scatterplot(data=embedding, x="x", y="y", ax=ax[0])
    ax[0].scatter(embedding_not_selected["x"], embedding_not_selected["y"],s=15, color=(0,0,1,0.1))
    ax[0].scatter(embedding_selected["x"], embedding_selected["y"],s=30, color=(1,0,0,1))

    for i,long in enumerate(select_longitudes):
        longitude = embedding[embedding["x_orig"] == long].sort_values("y_orig")
        ax[0].plot(longitude["x"],longitude["y"],c=cmap1(i/20),alpha=0.3)
    ax[0].set_title("longitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap1),ax=ax[0])

    # cmap2 = sns.color_palette("crest_r", as_cmap=True)
    cmap2 = colormaps.get_cmap("winter")

    sns.scatterplot(data=embedding, x="x", y="y",  ax=ax[1])
    ax[1].scatter(embedding_not_selected["x"], embedding_not_selected["y"], s=15,color=(0,0,1,0.1))
    ax[1].scatter(embedding_selected["x"], embedding_selected["y"],s=30, color=(1,0,0,1))
    for i,lat in enumerate(select_latitudes):
        latitudes = embedding[embedding["y_orig"] == lat].sort_values("x_orig")
        ax[1].plot(latitudes["x"],latitudes["y"],c=cmap2(i/20),alpha=0.3)
    ax[1].set_title("latitudes")
    plt.colorbar(cm.ScalarMappable(cmap=cmap2),ax=ax[1])

    plt.savefig(title)
    plt.show()
    return
def viz_manifold(manifold_name,locfile,title=r"E:\流形研究\20260201名古屋选址流形\fig\log_revmask_thre15.png"):
    res, nodes = read_manifold(manifold_name , nc=2)
    res = np.array(res)
    embedding = pd.DataFrame({"cid": nodes, "x": res.T[0], "y": res.T[1]})
    embedding["x_orig"] = embedding["cid"].apply(
        lambda x: cid_to_xy(x, ynum=40)[0])
    embedding["y_orig"] = embedding["cid"].apply(
        lambda x: cid_to_xy(x, ynum=40)[1])
    embedding = pick_selection_points(locfile, embedding)  # 选址点位标记在embedding中
    x_orig_range = embedding["x_orig"].max() - embedding["x_orig"].min()
    y_orig_range = embedding["y_orig"].max() - embedding["y_orig"].min()
    select_longitudes = [embedding["x_orig"].min() + int(i * x_orig_range / 20) for i in range(20)]
    select_latitudes = [embedding["y_orig"].min() + int(i * y_orig_range / 20) for i in range(20)]

    draw_latlon_fold(embedding, select_longitudes, select_latitudes,
                     title=title)
    return
if __name__=='__main__':
    day = 0#选取轨迹日
    coverfile = r'E:\基础数据\轨迹\日本\cover_day{day}.csv'.format(day=day)
    k = 200#选址中的设施/门店数量
    nc=5
    locfile = r'E:\流形研究\20260204选址流形整理\loc\greedy_nodup_locs_k{k}.npy'.format(k=k)
    matfile = r'E:\流形研究\20260204选址流形整理\mat_log_plus_rev_sqrt{nc}.csv'.format(nc=nc)
    vizfile=r'E:\流形研究\20260204选址流形整理\fig\log_revmask_thre15.png'
    inf=100
    dichtitle=r'E:\流形研究\20260204选址流形整理\dich\dich.csv'
    embfile = r'E:\流形研究\20260204选址流形整理\emb\R0_bnc5_inf100_k30_nb1000_tc0.9\nc2_perp6_thre15_w00.3_niter50_kiter10_N4\res.csv'
    # #生成cover文件
    # cid_cover=traj_to_cover(scale=5,ynumls=40,output=True,day=day)
    #
    # #选址
    # cover = read_cover(coverfile)
    # locs, nums = solve_greed_nodup(cover, k, locfile)

    # #生成距离矩阵
    #log_plus_mat_perp(coverfile, matfile, minlam=5, inf=inf, perp=6, opt='sqrt',nc=nc)

    # #嵌入
    R = 0
    para_res = {}
    perpsne = 30#早期版本用tsne可视化，该参数不影响tcie的嵌入，可无视
    nc = 5#维数，选址流形可视化须在二维，为保证嵌入质量可使用五维，即文中证明的维数
    inf = 100#替代无连接/非局部边的、理论应为无穷大的距离，无须调整
    niter = 50#用于tcie嵌入的收敛，下同，按照经验无须调整
    kiter = 10#同上
    N = 4#同上
    k_tcie = 30#tcie识别边界点的参数，无须调整
    tc = 0.9#同上
    perp = 6#局部的大小，可固定为该值
    para_res0 = {}#用于记录不同参数组合的嵌入效果
    for bnc in [5]:#局部维数，可固定
        for nb in [1000]:#边界点数量，可设置为大约所有节点数*0.9
            for thre in [1]:  #调参主要对象，在测度嵌入和拓扑嵌入之间权衡，
            #不同于实证部分（thre以下的边赋权w0），选址流形将thre以上的边赋权w0，因此降低thre在一定范围内会提升拓扑嵌入质量，降低测度嵌入质量
                for w0 in [0.3]:#按照经验，无须调整
                    r, yinter = tcie_para(R, perp, perpsne, bnc, nc, inf, k_tcie, nb, tc, thre, w0, niter, kiter, N,
                                          perptest=perp, show=False,
                                          matfile=matfile,
                                          dir0=r'E:\流形研究\20260204选址流形整理\emb_nc{nc}'.format(nc=nc), emb=True)
                    print(r, yinter)
                    para_res[(bnc, nb, thre, w0)] = r + yinter
                    para_res0[(bnc, nb, thre, w0)] = (r, yinter)
    print(para_res)
    temp = sorted(para_res.items(), key=lambda x: x[1], reverse=True)
    print(temp[0])
    print(para_res0)

    # #可视化

    # viz_manifold(embfile, locfile, title=vizfile)

    #二分分析
    # c_xy={}
    # res, nodes = read_manifold(embfile, nc=2)
    # for cid in nodes:
    #     c_xy[cid]=cid_to_xy(cid)
    # auto_dich(matfile, inf, r'E:\流形研究\20260204选址流形整理\dich', c_xy, dichthre=(0.5, 0.5), k=2,cut=False)



