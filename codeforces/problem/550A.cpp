#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace std;

const int N = 1e5 + 10;
char s[N];

int main() {
    cin.tie(0)->sync_with_stdio(0);
    cin >> s + 1;
    vector<int> a, b;
    for (int i = 1; s[i + 1]; ++ i) {
        if (s[i] == 'A' && s[i + 1] == 'B') a.push_back(i);
        if (s[i] == 'B' && s[i + 1] == 'A') b.push_back(i);
    }
    for (auto& i: a) {
        for (auto& j: b) {
            if (abs(i - j) != 1) {
                cout << "YES\n";
                return 0;
            }
        }
    }
    cout << "NO\n";
}