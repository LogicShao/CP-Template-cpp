#include <bits/stdc++.h>
using namespace std;
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n; cin >> n;
        vector<int> a(n, -1);
        int m = sqrt((n - 1) * 2);
        bool flag = true;
        for (int i = n - 1; ~i && flag; -- i)
            if (a[i] == -1) {
                for (; m * m >= i && a[m * m - i] != -1; -- m);
                int to = m * m - i;
                if (a[to] != -1) flag = false;
                a[to] = i;
                a[i] = to;
            }
        if (flag) for (auto &i: a) cout << i << ' ';
        else cout << -1;
        cout << endl;
    }
}