#include <bits/stdc++.h>
using namespace std;
typedef pair<char, int> P;
int main() {
    cin.tie(0)->sync_with_stdio(0);
    int T; cin >> T;
    while (T --) {
        string s; int p;
        cin >> s >> p;
        vector<P> v;
        for (int i = 0; i < s.size(); ++ i)
            v.push_back({s[i], i});
        sort(v.begin(), v.end());
        int val = 0;
        for (auto c: s) val += c - 'a' + 1;
        while (v.size() && val > p) {
            val -= v.back().first - 'a' + 1;
            v.erase(v.end());
        }
        sort(v.begin(), v.end(), [&](P a, P b) -> bool {
            return a.second < b.second;
        });
        for (auto [c, x]: v) cout << c;
        cout << endl;
    }
    return 0;
}
