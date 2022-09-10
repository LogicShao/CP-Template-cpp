#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef long long LL;

const int N = 2e5 + 10;
LL a[N], b[N];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int n, q; cin >> n >> q;
    for (int i = 1; i <= n; ++ i) cin >> a[i];
    for (int i = 1; i <= q; ++ i) {
        int l, r; cin >> l >> r;
        ++ b[l], -- b[r + 1];
    }
    for (int i = 1; i <= n; ++ i) b[i] += b[i - 1];
    sort(a + 1, a + 1 + n);
    sort(b + 1, b + 1 + n);
    LL res = 0;
    for (int i = 1; i <= n; ++ i) res += b[i] * a[i];
    cout << res << '\n';
}