#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <iomanip> // Để sử dụng setprecision
using namespace std;

// BFS tuần tự cho connected component
vector<int> bfs(const vector<vector<int>>& adj) {
    int V = adj.size();
    vector<bool> visited(V, false);
    vector<int> res;
    
    queue<int> q;
    
    int src = 0;
    visited[src] = true;
    q.push(src);

    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        res.push_back(curr);

        // Duyệt tất cả neighbor chưa visited
        for (int x : adj[curr]) {
            if (!visited[x]) {
                visited[x] = true;
                q.push(x);
            }
        }
    }
    
    return res;
}

// BFS song song sử dụng OpenMP (level-synchronous)
vector<int> parallel_bfs(const vector<vector<int>>& adj) {
    int V = adj.size();
    vector<char> visited(V, 0); // Sử dụng char để tránh vấn đề với vector<bool> trong multi-thread
    vector<int> res;
    
    int src = 0;
    visited[src] = 1;
    res.push_back(src);
    
    vector<int> current_level {src};
    
    while (!current_level.empty()) {
        vector<int> next_level;
        
        #pragma omp parallel
        {
            vector<int> local_next;
            
            #pragma omp for
            for (size_t i = 0; i < current_level.size(); ++i) {
                int curr = current_level[i];
                
                for (int x : adj[curr]) {
                    if (visited[x] == 0) {
                        #pragma omp critical
                        {
                            if (visited[x] == 0) {
                                visited[x] = 1;
                                local_next.push_back(x);
                            }
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                next_level.insert(next_level.end(), local_next.begin(), local_next.end());
            }
        }
        
        current_level = next_level;
        // Thêm level mới vào res (thứ tự trong level có thể khác tuần tự do parallel)
        res.insert(res.end(), current_level.begin(), current_level.end());
    }
    
    return res;
}

void addEdge(vector<vector<int>>& adj, int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
}

int main() {
    int levels = 20;
    int V = (1 << levels) - 1; // ≈1.048.575 đỉnh, đủ lớn để so sánh thời gian
    vector<vector<int>> adj(V);
    
    // Tạo đồ thị cây nhị phân hoàn chỉnh (undirected) - đầu vào giống nhau cho cả hai
    for (int i = 1; i < V; i++) {
        int parent = (i - 1) / 2;
        addEdge(adj, parent, i);
    }

    // Chạy BFS tuần tự
    double start_seq = omp_get_wtime();
    vector<int> res_seq = bfs(adj);
    double end_seq = omp_get_wtime();
    double time_seq = end_seq - start_seq;

    cout << fixed << setprecision(4); // Hiển thị với 2 chữ số thập phân
    cout << "Thoi gian hoan thanh (tuan tu): " << time_seq << " giay" << endl;
    cout << "Kich thuoc res (tuan tu): " << res_seq.size() << endl;

    // Chạy BFS song song
    double start_par = omp_get_wtime();
    vector<int> res_par = parallel_bfs(adj);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;

    cout << "Thoi gian hoan thanh (song song): " << time_par << " giay" << endl;
    cout << "Kich thuoc res (song song): " << res_par.size() << endl;

    return 0;
}