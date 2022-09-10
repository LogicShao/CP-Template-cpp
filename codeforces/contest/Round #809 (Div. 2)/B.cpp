#include <bits/stdc++.h>
using namespace std;
typedef pair<int, int> PII;
#define toMax(a, b) a < b ? a = b: 0
int main() {
    cin.tie()->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        int n; cin >> n;
        vector<PII> b(n + 1);
        for (int i = 1; i <= n; ++ i) {
            int x; cin >> x;
            b[i] = { x, i };
        }
        sort(b.begin() + 1, b.begin() + 1 + n);
        vector<int> dp(n + 1); // 颜色为i能放下的最大的塔
        int oddmax = 0, evenmax = 0; // 奇，偶位置放在最上面的最大值
        b[0] = { 0, 0 };
        for (int i = 1; i <= n; ++ i) {
            if (b[i].first != b[i - 1].first) {
                oddmax = 0, evenmax = 0;
            }
            int col = b[i].first, p = b[i].second;
            if (p & 1) { // 奇数位置应该放在偶数之上
                toMax(dp[col], evenmax + 1);
                toMax(oddmax, evenmax + 1);
            } else { // 偶数位置应该放在奇数之上
                toMax(dp[col], oddmax + 1);
                toMax(evenmax, oddmax + 1);
            }
        }
        for (int i = 1; i <= n; ++ i)
            cout << dp[i] << " \n"[i == n];
    }
}