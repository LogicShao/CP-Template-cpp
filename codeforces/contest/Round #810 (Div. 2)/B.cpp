#include <iostream>
#include <cstring>
#include <algorithm>
#define toMin(a, b) a > b ? a = b: 0

using namespace std;

typedef pair<int, int> PII;

const int N = 1e5 + 10;
int w[N], d[N];
PII e[N];
int n, m;

template<typename T> void read(T &x) {
    char ch = getchar(); bool flag = 0; x = 0;
    for (; ch < '0' || ch > '9'; ch = getchar())
        flag |= (ch == '-');
    for (; ch >= '0' && ch <= '9'; ch = getchar())
        x = (x << 1) + (x << 3) + (ch ^ 48);
    if (flag) x = -x;
}

int main() {
    int T; read(T);
    while (T --) {
        memset(d, 0, sizeof d);
        read(n), read(m);
        for (int i = 1; i <= n; ++ i) read(w[i]);
        for (int i = 1; i <= m; ++ i) {
            int a, b;
            read(a), read(b);
            e[i] = { a, b };
            ++ d[a], ++ d[b];
        }

        if (m & 1) {
            int res = 2e9;
            for (int i = 1; i <= n; ++ i)
                if (d[i] & 1)
                    toMin(res, w[i]);
            for (int i = 1; i <= m; ++ i) {
                auto [a, b] = e[i];
                if (d[a] + d[b] - 1 & 1)
                    toMin(res, w[a] + w[b]);
            }
            printf("%d\n", res);
        } else puts("0");
    }
}