# 搜索



## DFS

***

### 枚举

#### 指数型

```cpp
void calc(int x) {
    if (x > n) {
        for (int i: choose) cout << i << ' ';
        cout << endl;
        return;
    }
    calc(x + 1); // choose
    choose.push_back(x); // not choose
    calc(x + 1);
    choose.pop_back();
};
```

#### 组合型

```cpp
void calc(int x) {
    if (choose.size() > m || choose.size() + n - x + 1 < m)
        return;
    if (x > n) {
        for (int i: choose) cout << i << ' ';
        cout << endl;
        return;
    }
    choose.push_back(x); // not choose
    calc(x + 1);
    choose.pop_back();
    calc(x + 1); // choose
};
```

#### 排列型

```cpp
void calc(int x) {
    if (x > n) {
        for (int i = 1; i <= n; ++ i)
            printf("%d%c", a[i], " \n"[i == n]);
        return;
    }
    for (int i = 1; i <= n; ++ i)
        if (!st[i]) {
            a[x] = i, st[i] = true;
            calc(x + 1);
            a[x] = 0, st[i] = false;
        }
}
```



## BFS

***

### Flood-Fill

可以在线性时间复杂度内, 找到某个点所在的连通块.

[1097. 池塘计数 - AcWing题库](https://www.acwing.com/problem/content/1099/)

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 1005, M = N * N;
char g[N][N];
bool st[N][N];
int n, m, cnt;
PII q[M];

void bfs(int x, int y) {
    int hh = 0, tt = 0;
    q[tt] = {x, y};
    st[x][y] = true;
    
    while (hh <= tt) {
        auto [x, y] = q[hh ++];
        for (int i = x - 1; i <= x + 1; ++ i)
            for (int j = y - 1; j <= y + 1; ++ j) {
                if (i == x && j == y) continue;
                if (i < 0 || i >= n || j < 0 || j >= m) continue;
                if (g[i][j] == '.' || st[i][j]) continue;
                
                q[++ tt] = {i, j};
                st[i][j] = true;
            }
    }
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    
    cin >> n >> m;
    for (int i = 0; i < n; ++ i) cin >> g[i];
    
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < m; ++ j)
            if (g[i][j] == 'W' && !st[i][j]) {
                bfs(i, j);
                ++ cnt;
            }
    
    cout << cnt << '\n';
}
```

### 最短距离

要满足所有边的权重都相同

[1076. 迷宫问题 - AcWing题库](https://www.acwing.com/problem/content/description/1078/)

```cpp
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

#define x first
#define y second

using namespace std;

typedef pair<int, int> PII;

const int N = 1005;
const int dx[] = {0, -1, 0, 1};
const int dy[] = {-1, 0, 1, 0};
bool g[N][N];
bool st[N][N];
PII pre[N][N];
int n, idx;
PII q[N * N];

bool isin(int x, int y) {
    return 0 <= x && x < n && 0 <= y && y < n;
}

void bfs(int x, int y) {
    int hh = 0, tt = 0;
    q[0] = {x, y};
    st[x][y] = true;
    
    while (hh <= tt) {
        auto [x, y] = q[hh ++];
        if (x == n - 1 && y == n - 1) return;
        
        for (int i = 0; i < 4; ++ i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (isin(nx, ny) && !g[nx][ny] && !st[nx][ny]) {
                q[++ tt] = {nx, ny};
                pre[nx][ny] = {x, y};
                st[nx][ny] = true;
            }
        }
    }
}

void printpath(int x, int y, int cnt) {
    if (!x && !y) { cout << "0 0\n"; return; }
    auto [nx, ny] = pre[x][y];
    printpath(nx, ny, cnt + 1);
    cout << x << ' ' << y << '\n';
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    
    cin >> n;
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < n; ++ j)
            cin >> g[i][j];
            
    bfs(0, 0);
    
    printpath(n - 1, n - 1, 0);
}
```

### 多源BFS

增加一个虚拟源点

[173. 矩阵距离 - AcWing题库](https://www.acwing.com/problem/content/175/)

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;

const int N = 1010;
const int dx[] = {0, -1, 0, 1};
const int dy[] = {-1, 0, 1, 0};
char g[N][N];
int dist[N][N];
int n, m;
PII q[N * N];

bool isin(int x, int y) {
    return 0 <= x && x < n && 0 <= y && y < m;
}

void bfs() {
    memset(dist, -1, sizeof dist);
    int hh = 0, tt = -1;

    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < m; ++ j)
            if (g[i][j] == '1') {
                q[++ tt] = {i, j};
                dist[i][j] = 0;
            }
    
    while (hh <= tt) {
        auto [x, y] = q[hh ++];
        int d = dist[x][y];
        
        for (int i = 0; i < 4; ++ i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (isin(nx, ny) && dist[nx][ny] == -1) {
                dist[nx][ny] = d + 1;
                q[++ tt] = {nx, ny};
            }
        }
    }
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    
    cin >> n >> m;
    for (int i = 0; i < n; ++ i) cin >> g[i];
    
    bfs();
    
    for (int i = 0; i < n; ++ i)
        for (int j = 0; j < m; ++ j)
            cout << dist[i][j] << " \n"[j == m - 1];
}
```



### 最小步数

[AcWing 1107. 魔板 - AcWing](https://www.acwing.com/activity/content/problem/content/1475/)

[AcWing 845. 八数码 - AcWing](https://www.acwing.com/activity/content/problem/content/908/)

```cpp
// https://www.acwing.com/activity/content/problem/content/1475/
#include <iostream>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <queue>

using namespace std;

string start = "12345678";
unordered_map<string, int> dist;
unordered_map<string, pair<char, string>> path;
const int change[3][8] = {
    {7, 6, 5, 4, 3, 2, 1, 0},
    {3, 0, 1, 2, 5, 6, 7, 4},
    {0, 6, 1, 3, 4, 2, 5, 7}
};

string move(string s, int k) {
    string res;
    for (int i = 0; i < 8; ++ i) res += s[change[k][i]];
    return res;
}

int bfs(string start, string end) {
    queue<string> q;
    q.push(start);
    dist[start] = 0;
    
    while (q.size()) {
        string t = q.front();
        int d = dist[t];
        q.pop();
        
        if (t == end) return d;
        
        for (int i = 0; i < 3; ++ i) {
            string u = move(t, i);
            if (!dist.count(u)) {
                q.push(u);
                dist[u] = d + 1;
                path[u] = {'A' + i, t};
            }
        }
    }
    
    return -1;
}

void printpath(string s) {
    if (s == start) return;
    auto [op, str] = path[s];
    printpath(str);
    cout << op;
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    
    string end;
    for (int i = 0; i < 8; ++ i) {
        int x; cin >> x;
        end += '0' + x;
    }
    
    cout << bfs(start, end) << '\n';
    printpath(end);
}
```

### 双端队列BFS

[AcWing 175. 电路维修 - AcWing](https://www.acwing.com/activity/content/problem/content/1476/)

```cpp
#include <iostream>     /*如果边权只有0和1就可以将bfs转化为dijkstra*/
#include <cstring>      /*在本题中将联通的两点看成边权为0*/
#include <deque>        /*如果两点中间的标准件需要旋转才能联通则看为1*/
#include <algorithm>    /*即转化为最短路问题*/
                        /*容易知道奇点无法到达所以在bfs时不会出现某个点被操作两次*/
using namespace std;    /*该题点的坐标和标准件坐标并不同,注意变换*/

typedef pair<int, int> PII;

const int N = 510;
char g[N][N];
int dist[N][N];
bool st[N][N];
int n, m;

int bfs() {
    memset(dist, 0x3f, sizeof dist);
    memset(st, false, sizeof st);
    deque<PII> q;
    q.push_back({0, 0});
    dist[0][0] = 0;

    char cs[] = "\\/\\/";
    int dx[] = {-1, 1, 1, -1}, dy[] = {-1, -1, 1, 1};
    int ix[] = {-1, 0, 0, -1}, iy[] = {-1, -1, 0, 0};
    
    while (q.size()) {
        auto [x, y] = q.front();
        int d = dist[x][y];
        q.pop_front();
        
        if (st[x][y]) continue;
        st[x][y] = true;
        
        for (int i = 0; i < 4; ++ i) {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx < 0 || nx > n || ny < 0 || ny > m) continue;
            
            int gx = x + ix[i], gy = y + iy[i];
            int w = g[gx][gy] != cs[i];
            
            if (d + w < dist[nx][ny]) {
                dist[nx][ny] = d + w;
                if (w) q.push_back({nx, ny});
                else q.push_front({nx, ny});
            }
        }
    }
    
    return dist[n][m];
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    
    int T; cin >> T;
    while (T --) {
        cin >> n >> m;
        for (int i = 0; i < n; ++ i) cin >> g[i];
        
        if (n + m & 1) cout << "NO SOLUTION\n";
        else cout << bfs() << '\n';
    }
}
```

