#include <bits/stdc++.h>
using namespace std;
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n; cin >> n;
        vector<int> a(n + 1);
        a[0] = -2e9;
        for (int i = 1; i <= n; ++ i) cin >> a[i];
        int t = 0, now = 1;
        for (int i = 1; i <= n; ++ i) {
            if (now && a[i] < a[i - 1]) ++ t, now ^= 1;
            else if (!now && a[i] > a[i - 1]) ++ t, now ^= 1;
        }
        if (t <= 1) cout << "YES\n";
        else cout << "NO\n";
    }
}