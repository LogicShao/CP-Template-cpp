#include <bits/stdc++.h>

using namespace std;

using namespace std;

const int N = 1e5 + 100;
int T, n, a[N], l, r, tmp[N];

void work() {
    for (int i = l; i < r; i++) tmp[i] = a[i + 1] - a[i];
    sort(tmp + l, tmp + r);
    for (int i = l; i < r; i++) a[i] = tmp[i];
    if (l > 1) l--; r--;
}

int main() {
    scanf("%d", &T);
    while (T--) {
        scanf("%d", &n);
        for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
        l = 1; r = n; while (l < r && !a[l] && !a[l + 1]) l++;
        for (int i = 1; i < n; i++) {
            work();
            while (l < r && !a[l] && !a[l + 1]) l++;
        }
        printf("%d\n", a[l]);
    }
    return 0;
}