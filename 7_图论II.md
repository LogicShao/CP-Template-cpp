# 图论II

## SPFA差分约束&判负环

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

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

int main() {
    memset(h, -1, sizeof h);

    scanf("%d%d", &n, &m);
    while (m --) {
        int t, a, b;
        scanf("%d%d%d", &t, &a, &b);
        if (t == 1) add(a, b, 0), add(b, a, 0);
        else if (t == 2) add(a, b, 1);
        else if (t == 3) add(b, a, 0);
        else if (t == 4) add(b, a, 1);
        else add(a, b, 0);
    }

    for (int i = 1; i <= n; ++ i) add(0, i, 1);

    if (!spfa(0)) puts("-1");
    else {
        LL res = 0;
        for (int i = 1; i <= n; ++ i) res += dist[i];
        printf("%lld\n", res);
    }
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

## 有向图强连通分量

### tarjan

tarjan之后不需要拓扑排序，按照强连通分量逆序即为一个拓扑序

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

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

int main() {
    memset(h, -1, sizeof h);
    scanf("%d%d", &n, &m);
    while (m --) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);
    }
    for (int i = 1; i <= n; ++ i)
        if (!dfn[i])
            tarjan(i);
    for (int i = 1; i <= n; ++ i)
        for (int j = h[i]; ~j; j = ne[j]) {
            int k = e[j];
            int a = id[i], b = id[k];
            if (a != b) dout[a] ++;
        }
    int zeros = 0, sum = 0;
    for (int i = 1; i <= scc_cnt; ++ i)
        if (!dout[i]) {
            zeros ++;
            sum += siz[i];
            if (zeros > 1) {
                sum = 0;
                break;
            }
        }
    printf("%d\n", sum);
}
```
