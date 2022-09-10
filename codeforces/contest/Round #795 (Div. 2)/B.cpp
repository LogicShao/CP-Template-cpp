#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 1e5 + 10;
int p[N];
PII a[N];

int main() {
    int T;
    scanf("%d", &T);
    while (T --) {
        int n;
        scanf("%d", &n);
        for (int i = 1, t; i <= n; ++ i) {
            scanf("%d", &t);
            a[i] = {t, i};
        }
        sort(a + 1, a + 1 + n);
        bool flag = true;
        for (int i = 1, j = 1; i <= n && flag; i = ++ j) {
            while (j + 1 <= n && a[j].first == a[j + 1].first) ++ j;
            if (j == i) flag = false;
            else {
                p[a[j].second] = a[i].second;
                for (int k = i; k < j; ++ k) p[a[k].second] = a[k + 1].second;
            }
        }
        if (!flag) puts("-1");
        else {
            for (int i = 1; i <= n; ++ i) printf("%d%c", p[i], " \n"[i == n]);
        }
    }
}