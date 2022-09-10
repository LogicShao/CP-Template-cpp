# 图论

## 存图

邻接表

```c++
int h[N], e[M], ne[M], idx;
void add(int a, int b) { // 加边
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}
```

## 拓扑排序

有向图才有拓扑序列（有向无环图又叫拓扑图）

性质：一个有向无环图一定至少存在一个入度为0的点

* 度数

`入度` : 有向图中某点作为图中边的终点的次数之和

`出度` : 顶点的出边条数称为该顶点的出度

```c++
bool topsort() {
    int tt = -1, hh = 0;
    for (int i = 1; i <= n; ++ i) if (!d[i]) q[++ tt] = i;
    while (hh <= tt) {
        int t = q[hh ++];
        for (int i = h[t]; i != -1; i = ne[i]) {
            int j = e[i];
            d[j] --;
            if (!d[j]) q[++ tt] = j;
        }
    }
    return tt == n - 1;
}
```

## 最短路

n：定点数；m：边数

### 单源最短路

#### 所有边权都是正数

##### 朴素Dijkstra算法

O(n^2)       (m, n <= 1e5；即稠密图)

```c++
#include <iostream>
#include <cstring>

using namespace std;

const int N = 510;
int n, m, a, b, c;
int g[N][N]; // 稠密图用邻接矩阵存
int dist[N]; // 距离起点的距离
bool st[N]; // 该点距离有无确定

int dijkstra() {
    memset(dist, 0x3f, sizeof dist); // 所有点的距离初始化为+∞
    dist[1] = 0;
    for (int i = 0; i < n - 1; ++ i) {
        int t = -1;
        // 在所有没有确定最短距离的点中找到最近的点t
        for (int j = 1; j <= n; ++ j)
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
        // 用t点更新其他点的最短路
        for (int j = 1; j <= n; ++ j)
            dist[j] = min(dist[j], dist[t] + g[t][j]);
        st[t] = true;
    }
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    while (m --) {
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = min(g[a][b], c);
        // 重边取最小即可
    }
    printf("%d\n", dijkstra());
    return 0;
}
```

##### 堆优化的Dijkstra算法

O(mlogn)    (m,n > 1e5；即稀疏图)

```c++
// 堆优化dijkstra算法 --- 稀疏图
#include <iostream>
#include <queue>
#include <cstring>

using namespace std;

typedef pair<int, int> PII;

const int N = 1e6 + 10;
int n, m, a, b, c;
int h[N], w[N], e[N], ne[N], idx; // 稀疏图用邻接表存
int dist[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++;
}

int dijkstra() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    // 定义小根堆 first---距离 second---点的编号
    heap.push({0, 1});
    while (heap.size()) {
        // 每次取出距离最短的点
        auto t = heap.top();
        heap.pop();
        int ver = t.second, distance = t.first;
        // 如果ver点的最短路已经确定那么continue
        if (st[ver]) continue;
        st[ver] = true;
        // 更新ver点连的边的最短路
        for (int i = h[ver]; i != -1; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[ver] + w[i]) {
                dist[j] = dist[ver] + w[i];
                heap.push({dist[j], j});
            }
        }
    }
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    while (m --) {
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
        // 邻接表不用考虑重边
    }
    printf("%d\n", dijkstra());
    return 0;
}
```

#### 存在负权边

##### Bellman-Ford

O(nm)

```c++
#include <iostream>
#include <cstring>

using namespace std;

const int N = 510, M = 1e4 + 10;
struct Edge {
    int a, b, c;
} edge[M];
int n, m, k, a, b, c;
int dist[N];
int last[N]; // 上次的最短路

void bellman_ford() {
   memset(dist, 0x3f, sizeof dist);
   dist[1] = 0;
   for (int i = 0; i < k; ++ i) { // 走k次
       memcpy(last, dist, sizeof dist); // 防止出现串联
       for (int j = 0; j < m; j ++) {
           auto e = edge[j];
           dist[e.b] = min(dist[e.b], last[e.a] + e.c);
           // 松弛操作
       } // 三角不等式---dist[b]<=dist[a]+c
   }
}

int main() {
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 0; i < m; ++ i) {
        scanf("%d%d%d", &a, &b, &c);
        edge[i] = {a, b, c};
    }
    bellman_ford();
    // 因为存在负权边可能存在dist[x]==INF-y(y比较小)
    // 所有只要判断>一个比较大的数即可
    if (dist[n] > 0x3f3f3f3f / 2) puts("impossible");
    else printf("%d\n", dist[n]);
    return 0;
}
```

##### SPFA

一般：O(nm)；最坏：O(nm)

```c++
#include <iostream>
#include <cstring>
#include <cstdio>
#include <queue>

using namespace std;

const int N = 1e6 + 10;
int n, m, a, b, c;
int h[N], e[N], ne[N], idx, w[N];
int dist[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++;
}

int spfa() {
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    queue<int> q;
    q.push(1);
    st[1] = true;
    while (q.size()) {
        int t = q.front();
        q.pop();
        st[t] = false;
        for (int i = h[t]; i != -1; i = ne[i]) {
            int j = e[i];
            if (dist[j] > dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                if (!st[j]) {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }
    return dist[n];
}

int main() {
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    while (m --) {
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }
    int t = spfa();
    if (t == 0x3f3f3f3f) puts("impossible");
    else printf("%d\n", t);
    return 0;
}
```

### 多源汇最短路

#### Floyd算法

O(n^3)

源点即起点

汇点即终点

```c++
#include <iostream>
#include <cstring>

using namespace std;

const int N = 210, INF = 1e9;
int n, m, Q;
int d[N][N];
int a, b, c;

void floyd() {
    for (int k = 1; k <= n; ++ k)
        for (int i = 1; i <= n; ++ i)
            for (int j = 1; j <= n; ++ j)
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}

int main() {
    scanf("%d%d%d", &n, &m, &Q);
    for (int i = 1; i <= n; ++ i)
        for (int j = 1; j <= n; ++ j)
            if (i == j) d[i][j] = 0;
            else d[i][j] = INF;
    while (m --) {
        scanf("%d%d%d", &a, &b, &c);
        d[a][b] = min(d[a][b], c);
    }
    floyd();
    while (Q --) {
        scanf("%d%d", &a, &b);
        int t = d[a][b];
        if (t > INF / 2) puts("impossible");
        else printf("%d\n", t);
    }
    return 0;
}
```



## 最小生成树

### 朴素版Prim

O(n^2)

```c++
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 510, INF = 0x3f3f3f3f;
int n, m;
int g[N][N];
int dist[N];
bool st[N]; // 表示一个点是否在集合里
int a, b, c;

int prim() {
    memset(dist, 0x3f, sizeof dist);
    // 初始化成+∞
    int res = 0;
    for (int i = 0; i < n; ++ i) {
        // 找到集合外距离最近的点
        int t = -1;
        for (int j = 1; j <= n; ++ j)
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
        // 如果不能到达集合说明不存在最小生成树
        if (i && dist[t] == INF) return INF;
        // 此句写在31行前避免自环
        if (i) res += dist[t];
        st[t] = true; // 将t加入集合
        // 用t更新其他点到集合的距离
        for (int j = 1; j <= n; ++ j) dist[j] = min(dist[j], g[t][j]);
    }
    
    return res;
}

int main() {
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    while (m --) {
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = g[b][a] = min(g[a][b], c);
    }
    int t = prim();
    if (t == INF) puts("impossible");
    else printf("%d\n", t);
}
```

### 堆优化Prim

O(mlogn)

```c++
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>

using namespace std;

typedef pair<int, int> PII;

const int N = 5e3 + 5, M = 4e5 + 5, INF = 0x3f3f3f3f;
int st[N], dist[N];
int h[N], e[M], ne[M], w[M], idx;
int n, m;

void add(int a, int b, int c) {
    e[++ idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx;
}

int prim() {
    priority_queue<PII, vector<PII>, greater<PII>> q;
    memset(dist, 0x3f, sizeof dist);
    int res = 0, cnt = 0;
    dist[1] = 0;
    q.push({dist[1], 1});

    while (q.size() && cnt < n) {
        int d = q.top().first, t = q.top().second;
        q.pop();
        if (st[t]) continue;
        cnt ++;
        res += d;
        st[t] = 1;
        for (int i = h[t]; i; i = ne[i]) {
            int j = e[i];
            if (w[i] < dist[j]) {
                dist[j] = w[i];
                q.push({dist[j], j});
            }
        }
    }

    return cnt == n ? res: INF;
}

int main() {    
    scanf("%d%d", &n, &m);
    while (m --) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
        add(b, a, c);
    }
    
    int t = prim();

    if (t == INF) puts("orz");
    else printf("%d", t);
}
```

### Kruskal

O(mlogm)

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1e6 + 10, M = 2e6 + 10, INF = 0x3f3f3f3f;
int n, m;
int p[N]; // 并查集数组

struct Edge {
    int a, b, w;
    bool operator < (const Edge &W) const {
        return w < W.w;
    } // 运算符重载，写了可以不用写cmp
} edges[M];

int find(int x) {
    return p[x] == x ? p[x] : p[x] = find(p[x]);
}

int kruskal() {
    sort(edges, edges + m);
    // 将所有点按权重从小到大排序 O(mlogm)
    // 排序算法常数小
    // 所以能跑的比较快
    for (int i = 1; i <= n; ++ i) p[i] = i;
    // 初始化并查集
    int res = 0, cnt = 0;
    // res表示生成树中所有边权之和
    // cnt表示加入了多少边
    for (int i = 0; i < m; ++ i) {
        // 枚举每条边
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;
        // 如果a，b不连通
        a = find(a), b = find(b);
        if (a != b) {
            p[a] = b; // 将这条边加入集合中
            res += w;
            cnt ++;
        }
    }
    // 如果有边未加入集合说明最小生成树不存在
    if (cnt < n - 1) return INF;
    return res;
}

int main() {
    scanf("%d%d", &n, &m);
    for (int i = 0; i < m; ++ i) {
        int a, b, w;
        scanf("%d%d%d", &a, &b, &w);
        edges[i] = {a, b, w};
    }
    int t = kruskal();
    if (t == INF) puts("impossible");
    else printf("%d\n", t);
}
```



## 二分图

二分图：节点由两个集合组成，且两个集合内部没有边的图

性质：一个图是二分图当且仅当图中不含奇数环

### 染色法

O(n+m)

```c++
#include <iostream>
#include <cstring>

using namespace std;

const int N = 1e6 + 10, M = 2e6 + 10;
int n, m;
int h[N], e[M], ne[M], idx;
int color[N];
int a, b;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

bool dfs(int u, int c) {
    color[u] = c;
    // 将u染成c颜色
    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (!color[j]) { // 如果没有染色就染色
            if (!dfs(j, 3 - c)) return false; // 1->2, 2->1
        } // 相邻两点颜色不能相同
        else if (color[j] == c) return false;
    }
    return true;
}

int main() {
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);
    while (m --) {
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a); // 无向图
    }
    bool flag = true;
    // 遍历所有点，如果i未染色，将其所在的连通块染色
    // 如果一个点染色，那么这个点所在的连通块的颜色就已确定
    for (int i = 1; i <= n && flag; ++ i)
        if (!color[i] && !dfs(i, 1)) flag = false; // 如果产生矛盾则不是二分图
    puts(flag ? "Yes" : "No");
}
```

### 匈牙利算法

O(mn), 实际运行时间一般远小于O(mn) [可能是线性的也说不定]

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
// y总的奇妙比喻（男生表示左边点，女生表示右边点）
const int N = 510, M = 1e6 + 10;
int n1, n2, m;
int h[N], e[M], ne[M], idx; // 数组越界可能发生任何错误，不只是段错误
int match[N]; // 记录每个女生配对的男生
bool st[N]; // 记录每个女生有没有遍历过
int a, b;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

bool find(int x) {
    for (int i = h[x]; i != -1; i = ne[i]) {
        int j = e[i];
        if (!st[j]) {
            st[j] = true;
            if (match[j] == 0 || find(match[j])) { // 如果女生没有配对或能匹配别的男生则匹配
                match[j] = x;
                return true;
            }
        }
    }
    return false;
}

int main() {
    scanf("%d%d%d", &n1, &n2, &m);
    memset(h, -1, sizeof h);
    while (m --) {
        scanf("%d%d", &a, &b);
        add(a, b); // 存男生指向女生即可
    }
    int res = 0;
    for (int i = 1; i <= n1; ++ i) {
        memset(st, false, sizeof st);
        if (find(i)) res ++;
    }
    printf("%d\n", res);
}
```

## SPFA差分约束与判负环

```cpp
const int N = 1e5 + 10, M = N * 3;
int n, m;
int h[N], e[M], ne[M], w[M], idx;
int dist[N], cnt[N], q[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, ne[idx] = h[a], w[idx] = c, h[a] = idx ++;
}

bool spfa(int s) {
    memset(dist, 0xcf, sizeof dist);
    int hh = 0, tt = 1;
    q[0] = s;
    dist[s] = 0;
    st[s] = true;

    while (hh != tt) {
        int t = q[-- tt];
        st[t] = false;
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (dist[j] < dist[t] + w[i]) {
                dist[j] = dist[t] + w[i];
                cnt[j] = cnt[t] + 1;
                if (cnt[j] >= n + 1) return false;
                if (!st[j]) {
                    q[tt ++] = j;
                    st[j] = true;
                }
            }
        }
    }

    return true;
}
```

## LCA

### 倍增

```cpp
#include <iostream>
#include <cstdio>
#include <cstring>

using namespace std;

const int N = 5e5 + 10, M = N << 1;
int n, m, s;
int x, y;
int h[N], e[M], ne[M], idx;
int d[N], f[N][22], lg[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void dfs(int now, int fa) {
    f[now][0] = fa, d[now] = d[fa] + 1;
    for (int i = 1; i <= lg[d[now]]; ++ i)
        f[now][i] = f[f[now][i - 1]][i - 1];

    for (int i = h[now]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa) dfs(j, now);
    }
}

int lca(int x, int y) {
    if (d[x] < d[y]) swap(x, y);
    while (d[x] > d[y])
        x = f[x][lg[d[x] - d[y]] - 1];

    if (x == y) return x;

    for (int k = lg[d[x]] - 1; k >= 0; -- k)
        if (f[x][k] != f[y][k])
            x = f[x][k], y = f[y][k];
    return f[x][0];
}

int main() {
    memset(h , -1, sizeof h);

    scanf("%d%d%d", &n, &m, &s);

    for (int i = 1; i <= n - 1; ++ i) {
        scanf("%d%d", &x, &y);
        add(x, y), add(y, x);
    }

    for (int i = 1; i <= n; ++ i)
        lg[i] = lg[i - 1] + (1 << lg[i - 1] == i);

    dfs(s, 0);

    while (m --) {
        scanf("%d%d", &x, &y);
        printf("%d\n",lca(x, y));
    }
}
```

### 树链剖分

```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 4e4 + 10, M = N << 1;
int h[N], e[M], ne[M], idx;
int fa[N], dep[N], siz[N], son[N], top[N], dfn[N], rnk[N], tot;
int n, m, s;

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void dfs1(int p, int depth, int father) {
    fa[p] = father;
    dep[p] = depth;
    siz[p] = 1;
    int mmax = -1;

    for (int i = h[p]; ~i; i = ne[i]) {
        int j = e[i];
        if (j == father) continue;
        dfs1(j, depth + 1, p);
        if (siz[j] > mmax) {
            mmax = siz[j];
            son[p] = j;
        }
        siz[p] += siz[j];
    }
}

void dfs2(int p, int tp) {
    top[p] = tp;
    dfn[p] = ++ tot;
    rnk[tot] = p;

    if (!son[p]) return;

    for (int i = h[p]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != fa[p]) {
            if (j != son[p]) dfs2(j, j);
            else dfs2(j, tp);
        }
    }
}

int lca(int x, int y) {
    while (top[x] != top[y]) {
        if (dep[top[x]] >= dep[top[y]]) x = fa[top[x]];
        else y = fa[top[y]];
    }
    return dep[x] < dep[y] ? x : y;
}

int main() {
    memset(h, -1, sizeof h);

    scanf("%d", &n);

    for (int i = 1; i <= n; ++ i) {
        int u, v;
        scanf("%d%d", &u, &v);
        if (~v) add(u, v), add(v, u);
        else s = u;
    }

    dfs1(s, 1, s);
    dfs2(s, s);

    scanf("%d", &m);
    while (m --) {
        int a, b;
        scanf("%d%d", &a, &b);
        int Lca = lca(a, b), ans = 0;
        if (Lca == a) ans = 1;
        else if (Lca == b) ans = 2;
        printf("%d\n", ans);
    }
}
```



## 有向图强连通分量SCC

tarjan之后不需要拓扑排序，按照强连通分量逆序即为一个拓扑序

```cpp
const int N = 1e4 + 10, M = 5e5 + 10;
int n, m;
int h[N], e[M], ne[M], idx;
int stk[N], top;
bool in_stk[N];
int dfn[N], low[N], timestamp;
int dout[N];
int id[N], scc_cnt, siz[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void tarjan(int u) {
    dfn[u] = low[u] = ++ timestamp;
    stk[++ top] = u, in_stk[u] = true;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j);
            low[u] = min(low[u], low[j]);
        }
        else if (in_stk[j]) low[u] = min(low[u], dfn[j]);
    }
    
    if (dfn[u] == low[u]) {
        ++ scc_cnt;
        int y;
        do {
            y = stk[top --];
            in_stk[y] = false;
            id[y] = scc_cnt;
            siz[scc_cnt] ++;
        } while (y != u);
    }
}
```



## 无向图的双连通分量

### 边双连通分量E-DCC

极大的不含有桥的一个连通区域

```cpp
int h[N], e[M], ne[M], idx;
int n, m;
int dfn[N], low[N], timestamp;
int stk[N], top;
int id[N], dcc_cnt;
bool is_bridge[M];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void tarjan(int u, int from) {
    dfn[u] = low[u] = ++ timestamp;
    stk[++ top] = u;
    
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j, i);
            low[u] = min(low[u], low[j]);
            if (dfn[u] < low[j])
                is_bridge[i] = is_bridge[i ^ 1] = true;
        } else if (i != (from ^ 1))
            low[u] = min(low[u], dfn[j]);
    }
    
    if (dfn[u] == low[u]) {
        ++ dcc_cnt;
        int y;
        do {
            y = stk[top --];
            id[y] = dcc_cnt;
        } while (u != y);
    }
}
```

### 点双连通分量V-DCC

极大的不含有割点的一个连通区域

```cpp
const int N = 1010, M = 1010;
int h[N], e[M], ne[M], idx;
int dfn[N], low[N], timestamp;
int n, m, dcc_cnt, root;
int stk[N], top;
vector<int> dcc[N];
bool cut[N]; /*判断i是否为割点*/

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void tarjan(int u) {
    dfn[u] = low[u] = ++ timestamp;
    stk[++ top] = u;
    
    if (u == root && h[u] == -1) {
        dcc[++ dcc_cnt].push_back(u);
        return;
    }
    
    int cnt = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (!dfn[j]) {
            tarjan(j);
            low[u] = min(low[u], low[j]);
            if (low[j] >= dfn[u]) {
                ++ cnt;
                if (u != root || cnt > 1) cut[u] = true;
                ++ dcc_cnt;
                int y;
                do {
                    y = stk[top --];
                    dcc[dcc_cnt].push_back(y);
                } while (y != j);
                dcc[dcc_cnt].push_back(u);
            }
        } else low[u] = min(low[u], dfn[j]);
    }
}
```



## 欧拉回路与欧拉路径

邻接表

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10, M = 4e5 + 10;
int h[N], e[M], ne[M], idx;
int n, m, type, din[N], dout[N];
int ans[M >> 1], cnt;
bool used[M];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

void dfs(int u) {
    for (int &i = h[u]; ~i;) { /*防止被自环图卡*/
        if (used[i]) {
            i = ne[i]; continue;
        }
        
        used[i] = true;
        if (type == 1) used[i ^ 1] = true;
        
        int t;
        if (type == 1) {
            t = i / 2 + 1;
            if (i & 1) t = -t;
        } else t = i + 1;
        
        int j = e[i];
        i = ne[i];
        dfs(j);
        
        ans[++ cnt] = t;
    }
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    memset(h, -1, sizeof h);
    
    cin >> type >> n >> m;
    for (int i = 1; i <= m; ++ i) {
        int a, b;
        cin >> a >> b;
        add(a, b);
        if (type == 1) add(b, a);
        ++ dout[a], ++ din[b];
    }

    if (type == 1) {
        for (int i = 1; i <= n; ++ i)
            if (din[i] + dout[i] & 1) {
                cout << "NO"; return 0;
            }
    }
    else {
        for (int i = 1; i <= n; ++ i)
            if (din[i] != dout[i]) {
                cout << "NO"; return 0;
            }
    }

    for (int i = 1; i <= n; ++ i)
        if (h[i] != -1) {
            dfs(i); break;
        }
    
    if (cnt < m) {
        cout << "NO"; return 0;
    }
    
    cout << "YES\n";
    for (int i = cnt; i; -- i) cout << ans[i] << ' ';
}
```

邻接矩阵

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510, M = 1110;
int n = 500, m;
int g[N][N];
int ans[M], cnt;
int d[N];

void dfs(int u) {
    for (int i = 1; i <= n; ++ i)
        if (g[u][i]) {
            -- g[u][i], -- g[i][u];
            dfs(i);
        }
    ans[++ cnt] = u;
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    cin >> m;
    while (m --) {
        int a, b;
        cin >> a >> b;
        ++ g[a][b], ++ g[b][a];
        ++ d[a], ++ d[b];
    }
    
    int start = 1;
    for (int i = 1; i <= n; ++ i)
        if (d[i] && d[i] & 1) {
            start = i; break;
        }
            
    dfs(start);
    
    for (int i = cnt; i; -- i) cout << ans[i] << '\n';
}
```



## 网络流初步

### EK求最大流
```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1000 + 10, M = 20000 + 10, INF = 1 << 29;
int n, m, S, T;
int h[N], e[M], ne[M], f[M], idx;
int q[N], d[N], pre[N];
bool st[N];

void add(int a, int b, int c) {
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++;
}

bool bfs() {
    int hh = 0, tt = -1;
    memset(st, false, sizeof st);
    q[++ tt] = S;
    st[S] = true;
    d[S] = INF;
    
    while (hh <= tt) {
        int t = q[hh ++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int ver = e[i];
            if (!st[ver] && f[i]) {
                st[ver] = true;
                d[ver] = min(d[t], f[i]);
                pre[ver] = i;
                if (ver == T) return true;
                q[++ tt] = ver;
            }
        }
    }
    return false;
}

int ek() {
    int maxflow = 0;
    while (bfs()) {
        maxflow += d[T];
        for (int i = T; i != S; i = e[pre[i] ^ 1]) {
            f[pre[i]] -= d[T];
            f[pre[i] ^ 1] += d[T];
        }
    }
    return maxflow;
}

int main() {
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m --) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }
    printf("%d\n", ek());
}
```

### dinic求最大流
```cpp
#include <bits/stdc++.h>

using namespace std;

const int N = 1e5 + 10, M = N << 1, INF = 0x3f3f3f3f;

int n, m, S, T;
int h[N], e[M], ne[M], f[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c) {
    e[idx] = b, ne[idx] = h[a], f[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], f[idx] = 0, h[b] = idx ++;
}

bool bfs() {
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt) {
        int t = q[hh ++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int ver = e[i];
            if (d[ver] == -1 && f[i]) {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit) {
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i]) {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i]) {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;    
        }
    }
    return flow;
}

int dinic() {
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main() {
    scanf("%d%d%d%d", &n, &m, &S, &T);
    memset(h, -1, sizeof h);
    while (m --) {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }
    printf("%d\n", dinic());
    return 0;
}
```

### 点分裂

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

#define debug

using namespace std;

const int N = 55 << 1, M = (N * N) << 1, INF = 0x3f3f3f3f;
int n, m, S, T;
int h[N], e[M], ne[M], f[M], fs[M], idx;
bool g[N][N];
int q[N], cur[N], d[N];

void add(int a, int b, int c) {
    e[idx] = b, ne[idx] = h[a], f[idx] = c, h[a] = idx ++;
    e[idx] = a, ne[idx] = h[b], f[idx] = 0, h[b] = idx ++;
}

bool bfs() {
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt) {
        int t = q[hh ++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int ver = e[i];
            if (d[ver] == -1 && f[i]) {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit) {
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i]) {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i]) {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;    
        }
    }
    return flow;
}

int dinic() {
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

void init() {
    memset(h, -1, sizeof h);
    memset(g, 0, sizeof g);
    memset(f, 0, sizeof f);
    idx = 0;
}

int main() {
    while (scanf("%d%d", &n, &m) != EOF) {
        init();

        for (int i = 1; i <= m; ++ i) {
            int a, b;
            scanf(" (%d,%d)", &a, &b);
            a ++, b ++;
            add(a + n, b, INF);
            add(b + n, a, INF);
            g[a][b] = g[b][a] = true;
        }
        for (int i = 1; i <= n; ++ i) add(i, i + n, 1);
        memcpy(fs, f, sizeof f);

        int res = n;
        for (S = n + 1; S <= 2 * n; ++ S)
            for (T = S - n + 1; T <= n; ++ T)
                if (!g[S - n][T]) {
                    memcpy(f, fs, sizeof f);
                    res = min(res, dinic());
                }

        printf("%d\n", res);
    }
}
```