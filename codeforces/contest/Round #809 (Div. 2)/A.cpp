#include <bits/stdc++.h>

using namespace std;

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n, m; cin >> n >> m;
        vector<int> res(m + 1, 1);
        for (int i = 1; i <= n; ++ i) {
            int x; cin >> x;
            int p1 = x, p2 = m + 1 - x;
            if (p1 > p2) swap(p1, p2);
            if (res[p1] == 1) res[p1] = 0;
            else res[p2] = 0;
        }
        for (int i = 1; i <= m; ++ i)
            cout << (res[i] ? 'B': 'A');
        cout << endl;
    }
}