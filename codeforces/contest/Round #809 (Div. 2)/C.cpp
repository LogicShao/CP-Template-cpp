#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n; cin >> n;
        vector<int> a(n + 1);
        vector<LL> b(n + 2, 0), c(n + 2, 0);
        for (int i = 1; i <= n; ++ i) cin >> a[i];
        for (int i = 2; i < n; i += 2) { // 计算前缀和
            int mh = max(a[i - 1], a[i + 1]);
            b[i] = max(0, mh + 1 - a[i]);
        }
        for (int i = 1; i <= n; ++ i) b[i] += b[i - 1];
        for (int i = n - 1; i > 1; i -= 2) { // 计算后缀和
            int mh = max(a[i - 1], a[i + 1]);
            c[i] = max(0, mh + 1 - a[i]);
        }
        for (int i = n; i; -- i) c[i] += c[i + 1];
        if (n & 1) cout << b[n - 1] << endl; // 奇数个答案已经确定
        else { // 偶数个时，答案形式为其中两个间隔为2其余间隔都为1
            LL res = 1ll << 62;
            for (int i = 1; i < n; ++ i) {
                res = min(b[i] + c[i + 1], res);
            }
            cout << res << endl;
        }
    }
}