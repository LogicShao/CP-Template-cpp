#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10, M = N << 1;
int n, m;
int h[N], e[M], ne[M], idx;
bool hascat[N];
int d[N];

void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++;
}

int dfs(int u, int from, int cats) {
    if (cats > m) return 0;
    if (d[u] == 1 && u != 1) return 1;
    int res = 0;
    for (int i = h[u]; ~i; i = ne[i]) {
        int j = e[i];
        if (j != from) res += dfs(j, u, hascat[j] ? cats + 1: 0);
    }
    return res;
}

int main() {
    // cin.tie(0)->sync_with_stdio(0); time: 78 ms
    memset(h, -1, sizeof h);

    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; ++ i) scanf("%d", &hascat[i]);
    for (int i = 1; i < n; ++ i) {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b), add(b, a);
        ++ d[a], ++ d[b];
    }

    printf("%d\n", dfs(1, -1, hascat[1]));
}