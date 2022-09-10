#include <bits/stdc++.h>

using namespace std;

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;

    while (T --) {
        int n, p; cin >> n >> p;
        vector<int> a(n + 1), s(n + 1);

        for (int i = 1; i <= n; ++ i) cin >> a[i];

        int val = 0;
        for (int i = n; i; -- i) {
            if (val < a[i]) {
                if (val < p) {
                    val ++;
                    s[i] = 1;
                } else s[i] = 0;
            } else s[i] = 1;
        }

        for (int i = 1; i <= n; ++ i) cout << s[i];
        cout << '\n';
    }
}