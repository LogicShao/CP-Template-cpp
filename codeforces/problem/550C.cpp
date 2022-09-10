#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

const int N = 105;
char s[N];
int a[N], n;

int dfs(int u, int num, int cnt) {
    if (cnt && num % 8 == 0) return num;
    if (u > n || cnt > 3) return -1;
    // select u
    int res = dfs(u + 1, num * 10 + a[u], cnt + 1);
    if (res != -1) return res;
    // not select u
    res = dfs(u + 1, num, cnt);
    return res;
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    cin >> s + 1;
    for (int i = 1; s[i]; ++ i) a[i] = s[i] - '0', ++ n;
    int ans = dfs(1, 0, 0);
    if (ans == -1) cout << "NO\n";
    else cout << "YES\n" << ans << '\n';
}