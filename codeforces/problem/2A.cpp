#include <iostream>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#define toMax(a, b) a < b ? a = b: 0

using namespace std;

const int N = 1e3 + 10;
string a[N];
int b[N];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    unordered_map<string, int> mp, flag;
    int n; cin >> n;
    for (int i = 1; i <= n; ++ i) {
        cin >> a[i] >> b[i];
        mp[a[i]] += b[i];
    }
    int maxval = -2e9;
    for (auto [name, val]: mp) toMax(maxval, val);
    for (auto [name, val]: mp)
        if (val == maxval)
            flag[name] = 1;
    mp.clear();
    for (int i = 1; i <= n; ++ i) {
        mp[a[i]] += b[i];
        if (mp[a[i]] >= maxval && flag[a[i]]) {
            cout << a[i] << '\n';
            return 0;
        }
    }
}