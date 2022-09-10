#include <bits/stdc++.h>
#define int long long

using namespace std;

bool f(int n, int a, int b) {
    if (a == 1) return (n - 1) % b == 0;
    for (int k = 1; k <= n; k *= a)
        if ((n - k) % b == 0 && n - k >= 0)
            return true;
    return false;
}

signed main() {
    cin.tie(0)->sync_with_stdio(0);

    int T; cin >> T;
    while (T --) {
        int n, a, b;
        cin >> n >> a >> b;
        cout << (f(n, a, b) ? "Yes": "No") << '\n';
    }
}