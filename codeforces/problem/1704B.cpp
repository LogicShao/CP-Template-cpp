#include <bits/stdc++.h>
#define toMax(a, b) a < b ? a = b: 0
#define toMin(a, b) a > b ? a = b: 0
using namespace std;
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n, x; cin >> n >> x;
        vector<int> a(n + 1);
        for (int i = 1; i <= n; ++ i) cin >> a[i];
        int ans = -1;
        for (int i = 1; i <= n; ++ i) {
            int mmin = a[i], mmax = a[i], j = i;
            for (; j <= n; ++ j) {
                toMax(mmax, a[j]);
                toMin(mmin, a[j]);
                if (mmax - mmin > x * 2) break;
            }
            ++ ans;
            i = j - 1;
        }
        cout << ans << endl;
    }
}