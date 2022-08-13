# 图论II



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

