# Questions

## 994. Rotting Oranges

```java
// BFS, O(MN) time and space
class Solution {
    public int orangesRotting(int[][] grid) {
        int cnt = 0;
        int minutes = 0; // store steps
        Queue<int[]> q = new LinkedList<>();
        // traversal through graph, find starting node, add to queue
        // use cnt to track remaining fresh oranges
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 2) {
                    q.offer(new int[]{i, j});
                } else if (grid[i][j] == 1) {
                    cnt++;
                }
            }
        }
        if (cnt == 0) {
            return minutes;
        }
        int[] xDirect = {1, 0, -1, 0};
        int[] yDirect = {0, 1, 0, -1};
        // BFS
        while (!q.isEmpty()) {
            int size = q.size();
            minutes++;
            for (int i = 0; i < size; i++) {
                int[] coor = q.poll();
                for (int j = 0; j < 4; j++) {
                    int x = coor[0] + xDirect[j];
                    int y = coor[1] + yDirect[j];
                    if (x < 0 || y < 0 || x >= grid.length
                        || y >= grid[0].length || grid[x][y] != 1) {
                        continue;
                    }
                    grid[x][y] = 2;
                    cnt--;
                    q.offer(new int[]{x, y});
                }
            }
        }
        return cnt == 0 ? minutes - 1 : -1;
    }
}
```

## 346. Moving Average from Data Stream

```java
class MovingAverage {
    Queue<Integer> q;
    int size;
    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        q = new LinkedList<>();
        this.size = size;
    }

    public double next(int val) {
        q.offer(val);
        if (q.size() > size) {
            q.poll();
        }
        double sum = 0.0;
        for (int i : q) {
            sum += i;
        }
        return sum / q.size();
    }
}
```

## 286. Walls and Gates

```java
/* Idea: BFS
traversal through matrix, add all the coordinate of gate to the queue.
start BFS, for each point head four directions(check boundaries)
Time complexity: O(MN), Let us start with the case with only one gate. The breadth-first search takes at most m \times nm×n steps to reach all rooms, therefore the time complexity is O(MN). But what if you are doing breadth-first search from k gates?
Once we set a room's distance, we are basically marking it as visited, which means each room is visited at most once. Therefore, the time complexity does not depend on the number of gates and is O(MN).
Space complexity : O(MN). The space complexity depends on the queue's size. We insert at most m \times nm×n points into the queue.
*/
class Solution {
    private static final int GATE = 0;
    private static final int EMPTY = Integer.MAX_VALUE;
    private static final int[] xDirect = {1, 0, -1, 0};
    private static final int[] yDirect = {0, 1, 0, -1};

    public void wallsAndGates(int[][] rooms) {
        int m = rooms.length;
        if (rooms == null || m == 0) {
            return;
        }
        int n = rooms[0].length;
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (rooms[i][j] == GATE) {
                    q.offer(new int[]{i, j});
                }
            }
        }

        // BFS
        while (!q.isEmpty()) {
            int[] coor = q.poll();
            int row = coor[0];
            int col = coor[1];
            for (int i = 0; i < 4; i++) {
                int r = row + xDirect[i];
                int c = col + yDirect[i];
                // check boundaries
                if (r < 0 || c < 0 || r >= m || c >= n || rooms[r][c] != EMPTY) {
                    continue;
                }
                rooms[r][c] = rooms[row][col] + 1;
                q.offer(new int[]{r, c});
            }
        }
    }
}
```  

## 622. Design Circular Queue

```java
class MyCircularQueue {

    int[] queue;
    int head;
    int tail;
    int size;
    /** Initialize your data structure here. Set the size of the queue to be k. */
    public MyCircularQueue(int k) {
        queue = new int[k];
        head = -1;
        tail = -1;
        size = k;
    }

    /** Insert an element into the circular queue. Return true if the operation is successful. */
    public boolean enQueue(int value) {
        if (isFull()) {
            return false;
        }
        if (isEmpty()) { // add first element
            head = 0;
        }
        tail = (tail + 1) % size;
        queue[tail] = value;
        return true;
    }

    /** Delete an element from the circular queue. Return true if the operation is successful. */
    public boolean deQueue() {
        if (isEmpty()) {
            return false;
        }
        if (head == tail) { // delete last element
            head = -1;
            tail = -1;
            return true;
        }
        head = (head + 1) % size;
        return true;
    }

    /** Get the front item from the queue. */
    public int Front() {
        if (isEmpty()) {
            return -1;
        }
        return queue[head];
    }

    /** Get the last item from the queue. */
    public int Rear() {
        if (isEmpty()) {
            return -1;
        }
        return queue[tail];
    }

    /** Checks whether the circular queue is empty or not. */
    public boolean isEmpty() {
        return head == -1;
    }

    /** Checks whether the circular queue is full or not. */
    public boolean isFull() {
        return (tail + 1) % size == head;
    }
}
```

## 200. Number of Island

```java
// BFS
class Solution {
    private static final int[] xDirect = {1, 0, -1, 0};
    private static final int[] yDirect = {0, 1, 0, -1};

    public int numIslands(char[][] grid) {
        int m = grid.length;
        if (grid == null || m == 0) {
            return 0;
        }
        int n = grid[0].length;
        int cnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    cnt++;
                    // BFS
                    grid[i][j] = '0';
                    Queue<int[]> q = new LinkedList<>();
                    q.offer(new int[]{i, j});
                    while (!q.isEmpty()) {
                        int[] coor = q.poll();
                        int row = coor[0];
                        int col = coor[1];
                        for (int k = 0; k < 4; k++) {
                            int r = row + xDirect[k];
                            int c = col + yDirect[k];
                            if (r < 0 || c < 0 || r >= m || c >= n || grid[r][c] != '1') {
                                continue;
                            }
                            grid[r][c] = '0';
                            q.offer(new int[]{r, c});
                        }
                    }
                }
            }
        }
        return cnt;
    }
}
```

```java
// DFS
class Solution {
    public int numIslands(char[][] grid) {
        int cnt = 0;
        int m = grid.length;
        if (grid == null || m == 0) {
            return 0;
        }
        int n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    cnt++;
                    dfs(grid, i, j, m, n);
                }
            }
        }
        return cnt;
    }

    private void dfs(char[][] grid, int i, int j, int m, int n) {
        // recursion exit
        if (i < 0 || j < 0 || i >= m || j >= n || grid[i][j] != '1') {
            return;
        }
        // update current position
        grid[i][j] = '0';
        dfs(grid, i + 1, j, m, n);
        dfs(grid, i - 1, j, m, n);
        dfs(grid, i, j + 1, m, n);
        dfs(grid, i, j - 1, m ,n);
    }
}
```  

## 305. Number of Islands II

```java

```  

## 752. Open the Lock

```java
/* BFS,
setup: one set for dead end, one set for visited, one queue for BFS
Time complexity: O(N^2 * A^N + D), we may visited every lock combination, A is number of digit in the String, N is the number of digit in the lock, D is the size of dead end
Space complexity: (A^N + D),  for the queue and set.
*/
public int openLock(String[] deadends, String target) {
    Set<String> set = new HashSet(Arrays.asList(deadends));

    Queue<String> q = new LinkedList<>();
    q.offer("0000");

    Set<String> visited = new HashSet<>();
    visited.add("0000");

    int turns = 0;
    while (!q.isEmpty()) {
        int size = q.size();
        for (int k = 0; k < size; k++) {
            String state = q.poll();
            if (state.equals(target)) {
                return turns;
            } else if (!set.contains(state)) {
                for (int i = 0; i < 4; i++) {
                    for (int j = -1; j < 2; j += 2) {
                        // generate new code
                        int y = ((state.charAt(i) - '0') + j + 10) % 10;
                        String newState = state.substring(0, i) + ("" + y) + state.substring(i + 1);
                        if (!visited.contains(newState)) {
                            q.offer(newState);
                            visited.add(newState);
                        }
                    }
                }
            }
        }
        turns++;
    }
    return -1;
}
```

## 279. Perfect Square

```java
// Idea: Start from node 0 in queue, and keep pushing in perfect square number + current value, once we reach number n.
public int numSquares(int n) {
    Queue<Integer> q = new LinkedList<>();
    Set<Integer> visited = new HashSet<>();
    // start from 0
    q.offer(0);
    visited.add(0);
    int res = 0;
    while (!q.isEmpty()) {
        int size = q.size();
        res++;
        for (int i = 0; i < size; i++) {
            int num = q.poll();
            for (int j = 1; j * j <= n; j++) {
                // add new square to current num
                int sum = num + j * j;
                if (sum == n) {
                    return res;
                } else if (sum > n) {
                    break;
                }
                if (!visited.contains(sum)){
                    q.offer(sum);
                    visited.add(sum);
                }
            }
        }
    }
    return res;
}
```

## 841. Keys and Rooms

```java
// Idea: using queue to traversal starting from room zero, in each room, we add the keys in the room only if the room have not been visited.
class Solution {
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> q = new LinkedList<>();
        q.offer(0);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int room = q.poll();
                visited.add(room);
                for (int j = 0; j < rooms.get(room).size(); j++) {
                    if (!visited.contains(rooms.get(room).get(j))) {
                        q.offer(rooms.get(room).get(j));
                    }
                }
            }
        }
        return visited.size() == rooms.size();
    }
}
```

## 542. 01 Matrix

```java
// O(MN) time and O(1) space
class Solution {
    public int[][] updateMatrix(int[][] matrix) {
        int[] xDirect = {1, 0, -1, 0};
        int[] yDirect = {0, 1, 0, -1};
        int m = matrix.length;
        if (m == 0) {
            return new int[][]{};
        }
        int n = matrix[0].length;
        Queue<int[]> q = new LinkedList<>();
        // start from every zero
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    q.offer(new int[]{i, j});
                } else {
                    matrix[i][j] = Integer.MAX_VALUE;
                }
            }
        }
        // update non-zero point with smallest distance
        while (!q.isEmpty()) {
            int[] point = q.poll();
            int row = point[0];
            int col = point[1];
            for (int i = 0; i < 4; i++) {
                int r = row + xDirect[i];
                int c = col + yDirect[i];
                // check boundaries and if the new point is the smallest distance to 0;
                if (r < 0 || c < 0 || r >= m || c >= n || matrix[r][c] <= matrix[row][col] + 1) {
                    continue;
                }
                q.offer(new int[]{r, c});
                matrix[r][c] = matrix[row][col] + 1;
            }
        }
        return matrix;
    }
}
```

## 733. Flood Fill

```java
// Easy BFS solution, set start point {sr, sc}, record the color on that point, traversal adjacent position, skip if over the boundary, not the old color or this position is already updated.
class Solution {
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int[] xDirect = {1, 0, -1, 0};
        int[] yDirect = {0, 1, 0, -1};
        int m = image.length;
        int n = image[0].length;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{sr, sc});
        // record the original color
        int oldColor = image[sr][sc];
        while (!q.isEmpty()) {
            int[] start = q.poll();
            int row = start[0];
            int col = start[1];
            image[row][col] = newColor;
            for (int i = 0; i < 4; i++) {
                int r = row + xDirect[i];
                int c = col + yDirect[i];
                if (r < 0 || c < 0 || r >= m || c >= n || image[r][c] != oldColor || image[r][c] == newColor) {
                    continue;
                }
                q.offer(new int[]{r, c});
            }
        }
        return image;
    }
}
```

## 695. Max Area of Island

```java
// O(MN) time and space
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int[] xDirect = {1, 0, -1, 0};
        int[] yDirect = {0, 1, 0, -1};
        int m = grid.length;
        if (m == 0) return 0;
        int n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    grid[i][j] = 0;
                    int cnt = 0;
                    Queue<int[]> q = new LinkedList<>();
                    q.offer(new int[]{i, j});
                    while (!q.isEmpty()) {
                        int size = q.size();
                        for (int k = 0; k < size; k++) {
                            int[] coor = q.poll();
                            cnt++;
                            for (int l = 0; l < 4; l++) {
                                int x = coor[0] + xDirect[l];
                                int y = coor[1] + yDirect[l];
                                if (x < 0 || y < 0 || x >= m || y >= n
                                    || grid[x][y] == 0) {
                                    continue;
                                }
                                grid[x][y] = 0;
                                q.offer(new int[]{x, y});
                            }
                        }
                    }
                    res = Math.max(res, cnt);
                }
            }
        }
        return res;
    }
}
```

## 207. Course Schedule

```java
// O(MN) time and space
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {

        if (numCourses == 0 || prerequisites == null) {
            return false;
        }
        int[] inDegree = new int[numCourses];
        for(int[] pre : prerequisites) {
            inDegree[pre[0]]++;
        }

        List<Integer> res = new LinkedList<>();
        Queue<Integer> q = new LinkedList<>();

        for (int i = 0; i < numCourses; i++) {
           if(inDegree[i] == 0) {
               q.offer(i);
           }
        }
        while(!q.isEmpty()) {
            int course = q.poll();
            res.add(course);
            for (int[] pre : prerequisites) {
                if (course == pre[1]) {
                    inDegree[pre[0]]--;
                    if (inDegree[pre[0]] == 0) {
                        q.offer(pre[0]);
                    }
                }
            }
        }
        return res.size() == numCourses;
    }
}
```  

## 210. Course Schedule II

```java
// BFS, find a path which all nodes are connected.
// Directed graph, calculate indegree of each node.
// Time: calculate indegree, O(n), find course with no pre O(m),  check rest course O(mn)
// sum it up is O(mn)
public int[] findOrder(int numCourses, int[][] prerequisites) {
    // corner case
    if (numCourses == 0) return null;
    // Convert graph presentation from edges to indegree of adjacent list.
    int[] indegree = new int[numCourses];
    int[] res = new int[numCourses];
    int index = 0; // index of res
    // calculate how many prerequisites are needed for each course
    for (int i = 0; i < prerequisites.length; i++) {
        indegree[prerequisites[i][0]]++;
    }

    Queue<Integer> queue = new LinkedList<>();
    // first, add all the courses which have no prerequisites
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            queue.offer(i);
            res[index] = i;
            index++;
        }
    }

    // check the rest courses
    while (!queue.isEmpty()) {
        int pre = queue.poll();
        // traversal all the prerequisites
        for (int i = 0; i < prerequisites.length; i++) {
            // this course is another course's prerequisite
            if (prerequisites[i][1] == pre) {
                indegree[prerequisites[i][0]]--;
                // no pre for this course any more
                if (indegree[prerequisites[i][0]] == 0) {
                    res[index] = prerequisites[i][0];
                    index++;
                    queue.offer(prerequisites[i][0]);
                }
            }
        }
    }

    return index == numCourses ? res : new int[0];
}
```  

## 997. Find the Town Judge

```java
class Solution {
    public int findJudge(int N, int[][] trust) {
        int[] indegree = new int[N + 1];
        int[] outdegree = new int[N + 1];
        for (int[] t : trust) {
            indegree[t[1]]++;
            outdegree[t[0]]++;
        }
        for (int i = 1; i < N + 1; i++) {
            if (indegree[i] == N - 1 && outdegree[i] == 0) {
                return i;
            }
        }
        return -1;
    }
}
```

## 1042. Flower Planting With No Adjacent

```java
class Solution {
    public int[] gardenNoAdj(int N, int[][] paths) {
        int[] res = new int[N];
        // Create graph via adjancy list
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int i = 0; i < N; i++) {
            map.put(i, new HashSet<>());
        }

        // add edge
        for (int[] path : paths) {
            int v1 = path[0] - 1;
            int v2 = path[1] - 1;
            map.get(v1).add(v2);
            map.get(v2).add(v1);
        }

        // for each garden
        for (int i = 0; i < N; i++) {
            int[] colors = new int[5];
            for (int adj : map.get(i)) { // get each garden adjancy garden
                colors[res[adj]] = 1; // mark assigned color to 1
            }
            for (int c = 4; c >= 1; c--) {
                if (colors[c] != 1) { // if c color is not assgined to any garden
                    res[i] = c; // assign it to garden i
                }
            }
        }
        return res;
    }
}
```  

## 959. Regions Cut By Slashes

```java
// Union Find, each cell divide into four parts, top 0, right 1, bottom 2, left 3, union those little parts together.
class Solution {
    int count, N;
    int[] f;
    public int regionsBySlashes(String[] grid) {
        N = grid.length;
        // N * N grid
        f = new int[N * N * 4];
        count = N * N * 4;
        for (int i = 0; i < f.length; i++) {
            f[i] = i;
        }
        // traversal graph
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i > 0) {
                    union(getIndex(i - 1, j, 2), getIndex(i, j, 0));
                }
                if (j > 0) {
                    union(getIndex(i, j - 1, 1), getIndex(i, j, 3));
                }
                if (grid[i].charAt(j) != '/') {
                    union(getIndex(i, j, 0), getIndex(i, j, 1));
                    union(getIndex(i, j, 2), getIndex(i, j, 3));
                }
                if (grid[i].charAt(j) != '\\') {
                    union(getIndex(i, j, 0), getIndex(i, j, 3));
                    union(getIndex(i, j, 1), getIndex(i, j, 2));
                }
            }
        }
        return count;
    }

    public int find(int x) {
        if (x != f[x]) {
            f[x] = find(f[x]); // path compression
        }
        return f[x];
    }
    public void union(int x, int y) {
        // find the parent and save it to f[x]
        x = find(x);
        y = find(y);
        if (x != y) {
            f[x] = y;
            count--;
        }
    }

    public int getIndex(int i, int j, int k) {
        return (i * N + j) * 4 + k;
    }
}
```  

## 93. Restore IP Addresses

```java
// dfs solution
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        helper(res, "", s, 4);
        return res;
    }
    // k, 剩余需要拼接的段数
    public void helper(List<String> res, String sb, String s, int k) {
        // 如果没有剩余拼接的或者没有可以拼接的string，返回
        if (s.isEmpty() || k == 0) {
            if (s.isEmpty() && k == 0) {
                // 忽略leading “.”
                res.add(sb.substring(1));
            }
            return;
        }
        // avoid leading zero
        for (int i = 1; i <= (s.charAt(0) == '0' ? 1 : 3) && i <= s.length(); i++) {
            String part = s.substring(0, i);
            if (Integer.valueOf(part) <= 255) {
                helper(res, sb + "." + part, s.substring(i), k - 1);
            }
        }
    }
}
```  

## 399. Evaluate Division

```java
// DFS
class Solution {
    public double[] calcEquation(List<List<String>> equations,
                                double[] values,
                                List<List<String>> queries) {
        // create directed graph via adjancy list
        // a - > (b -> value)
        Map<String, Map<String, Double>> graph = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String left = equations.get(i).get(0);
            String right = equations.get(i).get(1);
            double value = values[i];
            graph.putIfAbsent(left, new HashMap<>());
            graph.putIfAbsent(right, new HashMap<>());
            graph.get(left).put(right, value);
            graph.get(right).put(left, 1.0 / value);
        }

        double[] res = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String left = queries.get(i).get(0);
            String right = queries.get(i).get(1);
            if (!graph.containsKey(left) || !graph.containsKey(right)) { // no result
                res[i] = -1.0;
                continue;
            }
            Set<String> visited = new HashSet<>();
            res[i] = helper(graph, left, right, visited); // dfs
        }
        return res;
    }

    public double helper(Map<String, Map<String, Double>> graph,
                         String left, String right,
                         Set<String> visited) {
        if (left.equals(right)) return 1.0;
        visited.add(left);
        for (Map.Entry<String, Double> entry : graph.get(left).entrySet()) {
            String div = entry.getKey();
            if (visited.contains(div)) continue;
            double d = helper(graph, div, right, visited);
            if (d > 0) { // find path
                return d * graph.get(left).get(div);
            }
        }
        return -1.0;
    }
}
```

## 269. Alien Dictionary

```java
// First create graph, then use BFS, start from node with 0 indegree.
class Solution {
    public String alienOrder(String[] words) {
        if (words.length == 0) return "";
        // calculate indegree
        Map<Character, Integer> indegree = new HashMap<>();
        for (String word : words) {
            for (char c : word.toCharArray()) {
                indegree.put(c, 0);
            }
        }
        // create graph
        Map<Character, Set<Character>> graph = new HashMap<>();
        for (int i = 0; i < words.length - 1; i++) {
            for (int j = 0; j < words[i].length(); j++) {
                char c1 = words[i].charAt(j);
                char c2 = words[i + 1].charAt(j);
                if (c1 != c2) {
                    graph.putIfAbsent(c1, new HashSet<>());
                    if (graph.get(c1).add(c2)) {
                        indegree.put(c2, indegree.get(c2) + 1);
                    }
                    break;
                }
            }
        }

        String res= "";
        Queue<Character> q = new LinkedList<>();
        for (char c : indegree.keySet()) {
            if (indegree.get(c) == 0) {
                q.offer(c);
            }
        }
        // bfs
        while (!q.isEmpty()) {
            char c = q.poll();
            res += c;
            if (graph.containsKey(c)) {
                for (char ch : graph.get(c)) {
                    indegree.put(ch, indegree.get(ch) - 1);
                    if (indegree.get(ch) == 0) {
                        q.offer(ch);
                    }
                }
            }
        }
        return res.length() == indegree.size() ? res : "";
    }
}
```

***

## Backtracking

## 46. Permutation

```java
// 三个思路解决，backtracking
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        permute(nums, lists, new ArrayList<>(), new boolean[nums.length]);
        return lists;
    }
    private void permute(int[] nums, List<List<Integer>> lists,
                         List<Integer> list, boolean[] used) {
        // recursion exit
        if (list.size() == nums.length) {
            lists.add(new ArrayList(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            list.add(nums[i]);
            used[i] = true;
            permute(nums, lists, list, used);
            used[i] = false;
            list.remove(list.size() - 1); // remove the last digit
        }
    }
}
```

```java
// 交换法
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        permute(nums, 0, lists); // 从下标0开始组合
        return lists;
    }

    private void permute(int[] nums, int begin, List<List<Integer>> lists) {
        if (begin == nums.length) {
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < nums.length; i++) {
                list.add(nums[i]);
            }
            lists.add(list);
            return;
        }

        for (int i = begin; i < nums.length; i++) {
            swap(nums, i, begin);
            permute(nums, begin + 1, lists);
            swap(nums, i, begin);
        }
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

}
```

## 77. Combination

```java
// backtracking
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> lists = new ArrayList<>();
        int[] nums = new int[n];
        boolean[] used = new boolean[n];
        for (int i = 0; i < n; i++) {
            nums[i] = i + 1;
        }
        backtrack(lists, new ArrayList<>(), nums, used, k, 0);
        return lists;
    }

    public void backtrack(List<List<Integer>> lists, List<Integer> list,
                          int[] nums, boolean[] used, int k, int start) {
        if (list.size() == k) {
            // System.out.println("test");
            lists.add(new ArrayList(list));
            return;
        }
        for (int i = start; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            list.add(nums[i]);
            used[i] = true;
            backtrack(lists, list, nums, used, k, i + 1);
            used[i] = false;
            list.remove(list.size() - 1);
        }
    }
}
```

## 39. Combination Sum

```java
// Backtracking solution
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> lists = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(lists, new ArrayList<>(), candidates, target, 0);
        return lists;
    }

    private void backtrack(List<List<Integer>> lists, List<Integer> list, int[] nums, int remain, int start) {
        if (remain < 0) { // recursion exit
            return;
        } else if (remain == 0) { // find one solution
            lists.add(new ArrayList(list));
        } else {
            for (int i = start; i < nums.length; i++) { // 遍历candidates
                list.add(nums[i]);
                backtrack(lists, list, nums, remain - nums[i], i); // not i + 1, since num can be reused
                list.remove(list.size() - 1);
            }
        }
    }
}
```  

## 40. Combination Sum II

```java
// 思路同上，在遍历元素时，跳过重复项，递归时，不使用重复元素
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> lists = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(lists, new ArrayList(), candidates, target, 0);
        return lists;
    }

    private void backtrack(List<List<Integer>> lists, List<Integer> list, int[] nums, int remain, int start) {
        if (remain < 0) {
            return;
        } else if (remain == 0) {
            lists.add(new ArrayList(list));
        } else {
            for (int i = start; i < nums.length; i++) {
                if (i > start && i < nums.length && nums[i] == nums[i - 1]) { // skip duplicate
                    continue;
                }
                list.add(nums[i]);
                backtrack(lists, list, nums, remain - nums[i], i + 1); // no reuse of elements
                list.remove(list.size() - 1);
            }
        }
    }
}
```  

## 52. N-Queens

```java
// 思路同上，在遍历到每一行的时候新建一个String，加到board里，在backtrack的时候再将最后一位删除
class Solution {
    int[] col;
    int[] diag1;
    int[] diag2;
    List<String> board;
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        col = new int[n];
        diag1 = new int[2*n - 1];
        diag2 = new int[2*n - 1];
        board = new ArrayList<>();
        helper(n, 0, res);
        return res;
    }

    public void helper(int n, int row, List<List<String>> res) {
        if (row == n) {
            res.add(new ArrayList<>(board));
            return;
        }

        for (int c = 0; c < n; c++) {
             if (!available(row, c, n)) {
                continue;
            }
            char[] charArray = new char[n];
            Arrays.fill(charArray, '.');
            charArray[c] = 'Q';
            String rowString = new String(charArray);
            board.add(rowString);
            update(row, c, 1, n);
            helper(n, row + 1, res);
            board.remove(board.size() - 1);
            update(row, c, 0, n);
        }
    }

    public boolean available(int r, int c, int n) {
        return (col[c] == 0) && (diag1[r + c] == 0) && (diag2[c - r + n - 1] == 0);
    }

    public void update(int r, int c, int val, int n) {
        col[c] = val;
        diag1[r + c] = val;
        diag2[c - r + n - 1] = val;
    }
}
```  

## 53. N-Queens II

```java
// 递归查询，每放入一个queen，标记当前row, column和对角线
// row和column有index标记，对角线赋予index，/ 方向设为 x + y, \ 方向设为 x - y + (n - 1)
// 因为x - y 的范围是 -(n - 1) ~ (n - 1)，加上(n - 1)就变成了0 ~ 2n - 2
class Solution {
    int[] col; // 列
    int[] diag; // 对角线
    int[] rev_diag; // 反对角线
    int res;
    public int totalNQueens(int n) {
        col = new int[n];
        diag = new int[n*2 - 1];
        rev_diag = new int[n*2 - 1];
        helper(n, 0); // 从第0行开始递归调用
        return res;
    }

    public void helper(int n, int row) {
        if (row == n) {
            res++;
        }
        for (int c = 0; c < n; c++) { // 遍历col
            if (!available(row, c, n)) {
                continue;
            }
            update(row, c, 1, n);
            helper(n, row + 1);
            update(row, c, 0, n);
        }
    }

    public boolean available(int r, int c, int n) { // 检查当前的列和对角线
        return (col[c] == 0) && (diag[r + c] == 0) && (rev_diag[c - r + n - 1] == 0);
    }

    public void update(int r, int c, int val, int n) { // 更新列和对角线
        col[c] = val;
        diag[r + c] = val;
        rev_diag[c - r + n - 1] = val;
    }
}
```  

## 301. Remove Invalid Parentheses

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        dfs(res, s, 0, 0, '(', ')');
        return res;
    }

    private void dfs(List<String> res, String s, int iStart, int jStart, char open, char close) {
        // 计算'('和')'的数量
        int numOpen = 0, numClose = 0;
        // 我们从iStart开始遍历, 也是在删除多余')'后开始遍历的起点
        for (int i = iStart; i < s.length(); i++) {
            // 正向操作时，只处理')'多余的情况
            if (s.charAt(i) == open) numOpen++;
            if (s.charAt(i) == close) numClose++;
            if (numClose > numOpen) {
                // 移除多余的右括号, 可能有多种移除的方案，因此要遍历所有的右括号, 此时要移除的右括号一定在i的左边
                for (int j = jStart; j <= i; j++) {
                    // 移除时要考虑相邻的右括号重复的情况, (j == jStart)为删除第一个')'的情况
                    // s.charAt(j - 1) != close 考虑当前为')'同时前一个不能为')'
                    if (s.charAt(j) == close && (j == jStart || s.charAt(j - 1) != close)) {
                        // 删除下标为j的右括号后，判断是否valid, iStart = i因为该字符串到i的位置都已经判断过为valid
                        // jStart = j表示跳过了前一层循环结束时的末尾字符
                        dfs(res, s.substring(0, j) + s.substring(j + 1, s.length()), i, j, open, close);
                    }
                }
                return; // 递归操作处理后面的字符
            }
        }
        // 正向检查过一遍，接下来倒过来检查是否有多余的'('
        String reversed = new StringBuilder(s).reverse().toString();
        if (open == '(') { // 如果是正向到了这步，说明要检查反向
            dfs(res, reversed, 0, 0, ')', '(');
        } else { // 如果是反向到了这步，则把反向翻转过来，添加到结果中
            res.add(reversed);
        }
    }
}
```  

## 32. Longest Valid Parentheses

```java
// 思路，遍历一遍String，如果遇到（ 加到stack，如果遇到），判断stack是否为空
// stack空则直接加到stack，如果不为空，判断peek是否为（，如果是则pop更新res
// 否则入栈，每次入栈的元素是当前符号的下标
class Solution {
    public int longestValidParentheses(String s) {
        int res = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    if (s.charAt(stack.peek()) == '(') {
                        stack.pop();
                        res = Math.max(i - (stack.empty() ? -1 : stack.peek()), res);
                    } else {
                        stack.push(i);
                    }
                }
            }
        }
        return res;
    }
}

// DP solution
/*
If s[i] is '(', set longest[i] to 0,because any string end with '(' cannot be a valid one.
Else if s[i] is ')'
    If s[i-1] is '(', longest[i] = longest[i-2] + 2
    Else if s[i-1] is ')' and s[i-longest[i-1]-1] == '(', longest[i] = longest[i-1] + 2 + longest[i-longest[i-1]-2]

For example, input "()(())", at i = 5, longest array is [0,2,0,0,2,0], longest[5] = longest[4] + 2 + longest[1] = 6.
*/
class Solution {
    public int longestValidParentheses(String s) {
        int[] dp = new int[s.length()];
        int res = 0;
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                dp[i] = 0;
            } else {
                if (s.charAt(i - 1) == '(') {
                    if (i - 2 >= 0) {
                        dp[i] = dp[i - 2] + 2;
                    } else {
                        dp[i] = 2;
                    }
                    res = Math.max(dp[i], res);
                } else {
                    if (i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                        dp[i] = dp[i - 1] + 2;
                        if (i - dp[i - 1] - 2 >= 0) {
                            dp[i] = dp[i] + dp[i - dp[i - 1] - 2];
                        }
                        res = Math.max(dp[i], res);
                    }
                }
            }
        }
        return res;
    }
}
```  

## 489. Robot Room Cleaner

```java
// DFS + backtracking
// 以起点作为坐标轴原点，用set记录经过的点
class Solution {
    int[] xDirect = {1, 0, -1, 0};
    int[] yDirect = {0, 1, 0, -1};
    public void cleanRoom(Robot robot) {
        dfs(robot, new HashSet(), 0, 0, 0);
    }
    // x, y: 当前节点坐标
    // dir: 当前方向, 0: up, 1: right, 2: down, 3: left
    public void dfs(Robot robot, Set<String> visited, int x, int y, int dir) {
        String key = x + "," + y;
        if (visited.contains(key)) {
            return;
        }
        visited.add(key);
        robot.clean();
        for (int i = 0; i < 4; i++) {
            if (robot.move()) { // 如果可以沿着当前方向走
                dfs(robot, visited, x + xDirect[dir], y + yDirect[dir], dir);
                backtrack(robot);
            }
            // 如果当前方向不能走, 换方向继续走
            robot.turnRight();
            dir += 1;
            dir %= 4;
        }
    }
    // 往回走一格
    public void backtrack(Robot robot) {
        robot.turnRight();
        robot.turnRight();
        robot.move();
        robot.turnLeft();
        robot.turnLeft();
    }
}
```  

## 139. Word Break

BFS solution O(n^2)

```java
public boolean wordBreak(String s, List<String> wordDict) {
    Set<String> dict = new HashSet(wordDict);
    if (dict.contains(s)) {
        return true;
    }

    Queue<Integer> q = new LinkedList<>(); // index queue
    q.offer(0);
    Set<Integer> visited = new HashSet<Integer>();
    visited.add(0);
    while (!q.isEmpty()) {
        int idx = q.poll();
        for (int i = idx + 1; i <= s.length(); i++) {
            if (visited.contains(i)) {
                continue;
            }
            if (dict.contains(s.substring(idx, i))) {
                if (i == s.length()) {
                    return true;
                }
                q.offer(i);
                visited.add(i);
            }
        }
    }
    return false;
}
```  

DP solution O(n^2) time and O(n) space

```java
public boolean wordBreak(String s, List<String> wordDict) {
    Set<String> dict = new HashSet(wordDict);
    boolean[] dp = new boolean[s.length() + 1];
    dp[0] = true;
    for (int i = 1; i <= s.length(); i++) {
        for (int j = 0; j < i; j++) {
            if (dp[j] && dict.contains(s.substring(j, i))) {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[s.length()];
}
```  

## 79. Word Search

思路： DFS，遍历board，找到和word第一个字母相同时开始dfs。

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if (word == null || word.length() == 0 || board.length == 0) {
            return true;
        }
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == word.charAt(0) && dfs(board, word, visited, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(char[][] board, String word, boolean[][] visited,
                        int i, int j, int idx) {
        // the last character match
        if (idx == word.length()) {
            return true;
        }
        // index out of bound, character doesn't match or been visited
        if (i < 0 || i >= board.length
            || j < 0 || j >= board[i].length
            || board[i][j] != word.charAt(idx)
            || visited[i][j]) {
            return false;
        }
        visited[i][j] = true;
        // dfs all four directions
        if (dfs(board, word, visited, i + 1, j, idx + 1)
           || dfs(board, word, visited, i - 1, j, idx + 1)
           || dfs(board, word, visited, i, j + 1, idx + 1)
           || dfs(board, word, visited, i, j - 1, idx + 1)) {
            return true;
        }
        // backtracking, reset visited point
        visited[i][j] = false;
        return false;
    }
}
```  

## 212. Word Search II

```java
// 第一个思路，每个单词都用dfs在matrix中查找，O(MNN)
// 用trie，O(MNLogN)
class TrieNode {
    TrieNode[] children;
    String word;

    public TrieNode() {
        // a-z, all lower case
        children = new TrieNode[26];
    }
}
class Solution {
    public List<String> findWords(char[][] board, String[] words) {
        List<String> res = new ArrayList<>();
        // 建树，将单词存到树的叶子节点
        TrieNode root = buildTrie(words);
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, i, j, root, res);
            }
        }
        return res;
    }

    public void dfs(char[][] board, int i, int j, TrieNode runner, List<String> res) {
        char c = board[i][j]; // for backtracking
        if (c == '#' || runner.children[c - 'a'] == null) {
            return;
        }
        runner = runner.children[c - 'a'];
        if (runner.word != null) {
            res.add(runner.word);
            runner.word = null; // 将当前单词设为null，防止相同结果出现
        }
        board[i][j] = '#'; // 标记为当前单词已用过的点
        // 提前检查边界，减少递归调用
        if (i > 0) {
            dfs(board, i - 1, j, runner, res);
        }
        if (j > 0) {
            dfs(board, i, j - 1, runner, res);
        }
        if (i < board.length - 1) {
            dfs(board, i + 1, j, runner, res);
        }
        if (j < board[0].length - 1) {
            dfs(board, i, j + 1, runner, res);
        }
        board[i][j] = c;
    }

    public TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String word : words) {
            TrieNode runner = root;
            for (char c : word.toCharArray()) {
                if (runner.children[c - 'a'] == null) {
                    runner.children[c - 'a'] = new TrieNode();
                }
                runner = runner.children[c - 'a'];
            }
            runner.word = word;
        }
        return root;
    }
}
```  

## 127. Word Ladder

1. Bidirectional Searching, 用两个set，从两端开始搜索

```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    Set<String> dict = new HashSet(wordList);
    if (!dict.contains(endWord)) { // 不包含endWord，直接返回0
        return 0;
    }

    Set<String> beginSet = new HashSet<>();
    Set<String> endSet = new HashSet<>();
    beginSet.add(beginWord);
    endSet.add(endWord);

    int len = 1; // 路径长度
    Set<String> visited = new HashSet<>();
    visited.add(beginWord);
    visited.add(endWord);

    while (!beginSet.isEmpty() && !endSet.isEmpty()) {
        if (beginSet.size() > endSet.size()) { // 确保beginSet是更短的那一条路
            Set<String> set = beginSet;
            beginSet = endSet;
            endSet = set;
        }

        Set<String> tmp = new HashSet<>(); // 从beginWord开始到下一层所有节点可能
        for (String word : beginSet) {
            char[] chs = word.toCharArray();
            for (int i = 0; i < chs.length; i++) { // 遍历字母表，生成新的单词
                for (char c = 'a'; c <= 'z'; c++) {
                    char old = chs[i];
                    chs[i] = c;
                    String newWord = String.valueOf(chs);

                    if (endSet.contains(newWord)) { // endSet包含单词，说明找了一条路径
                        return len + 1;
                    }

                    if (dict.contains(newWord) && !visited.contains(newWord)) { // 避免重复遍历
                        tmp.add(newWord);
                        visited.add(newWord);
                    }
                    chs[i] = old; // 还原单词
                }
            }
        }
        beginSet = tmp; // 更新下一层需要遍历的单词
        len++;
    }
    return 0;
}
```  

## 1192. Critical Connections in a Network

```java
// Tarjan Algorithm
// O(V + E)
class Solution {
    // Basically, it uses dfs to travel through the graph to find if current vertex u,
    // can travel back to u or previous vertex
    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        int[] lowLinkValue = new int[n]; // records the lowest vertex u can reach
        int[] nodes = new int[n]; // records time when a node is visited
        Arrays.fill(nodes, -1); // -1 represent the node has not been visited
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>(); // initialization
        }
        // build graph
        for (List<Integer> conn : connections) {
            int src = conn.get(0);
            int tar = conn.get(1);
            graph[src].add(tar);
            graph[tar].add(src);
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (nodes[i] == -1) {
                dfs(i, lowLinkValue, nodes, graph, res, i);
            }
        }
        return res;
    }

    int time = 0;
    public void dfs(int u, int[] low, int[] nodes, List<Integer>[] graph, List<List<Integer>> res, int pre) {
        nodes[u] = time;
        low[u] = time;
        time++;
        for (int i = 0; i < graph[u].size(); i++) { // 以u为起点
            int v = graph[u].get(i);
            if (v == pre) { // ignore parent node
                continue;
            }
            if (nodes[v] == -1) { // v 没有遇到过
                dfs(v, low, nodes, graph, res, u);
                low[u] = Math.min(low[u], low[v]);
                // 如果是强连通图，且u,v相连，low[v] == nodes[u]
                // 如果两者不属于同一个强连通图，low[v]就不会更新，因此会大于nodes[u]
                if (low[v] > nodes[u]) {
                    // u - v is critical, there is no path for v to reach back to u
                    // or previous vertices of u
                    res.add(Arrays.asList(u, v));
                }
            } else {
                // if v discovered and is not parent of u,
                // update low[u], cannot use low[v] because u is not subtree of v
                low[u] = Math.min(low[u], nodes[v]);
            }
        }
    }
}
```  

## 332. Reconstruct Itinerary

Eulerian Path, 有向图寻找欧拉路径, 解决欧拉路径的算法，Hierholzer.  

```java
path = []
DFS(u):
    While (u存在未被访问的边e(u,v))
        mark边e(u,v)为访问
        DFS(v)
    End
    path.pushLeft(u)
```

``` java
class Solution {
    Map<String, PriorityQueue<String>> flights; // edge start -> edge end
    LinkedList<String> path;

    public List<String> findItinerary(List<List<String>> tickets) {
        flights = new HashMap<>();
        path = new LinkedList<>();
        for (int i = 0; i < tickets.size(); i++) { // 遍历所有边
            flights.putIfAbsent(tickets.get(i).get(0), new PriorityQueue<>());
            flights.get(tickets.get(i).get(0)).add(tickets.get(i).get(1));
        }

        dfs("JFK");
        return path;
    }

    public void dfs(String departure) {
        PriorityQueue<String> arrivals = flights.get(departure); // pq为升序排列，每次取到的都是lexical 小的结果
        while (arrivals != null && !arrivals.isEmpty()) { // Hierholzer算法
            dfs(arrivals.poll());
        }
        path.addFirst(departure);
    }
}
```  

## 547. Friend Circle

```java
class Solution {
    public int findCircleNum(int[][] M) {
        int circle = 0;
        boolean[] visited = new boolean[M.length]; // 下标对应学生
        for (int i = 0; i < M.length; i++) { // 遍历所有学生
            if (!visited[i]) { // A学生还没有加入circle
                dfs(M, visited, i);
                circle++;
            }
        }
        return circle;
    }

    public void dfs(int[][] m, boolean[] visited, int i) {
        for (int j = 0; j < m.length; j++) { // 遍历A学生的朋友关系
            if (m[i][j] == 1 && !visited[j]) { // 如果A和B有朋友关系, 并且B没在circle里，如果在会死循环
                visited[j] = true; // B加入到circle
                dfs(m, visited, j); // 遍历B的除去A的朋友
            }
        }
    }
}
```  

## 339. Nested List Weight Sum

```java
// recursion
class Solution {
    public int depthSum(List<NestedInteger> nestedList) {
        return dfs(nestedList, 1);
    }

    public int dfs(List<NestedInteger> list, int depth) {
        int sum = 0;
        for (NestedInteger n : list) {
            if (n.isInteger()) {
                sum += depth * n.getInteger();
            } else {
                sum += dfs(n.getList(), depth + 1);
            }
        }
        return sum;
    }
}
```  

```java
// Iteration
class Solution {
    public int depthSum(List<NestedInteger> nestedList) {
        int sum = 0;
        int depth = 1;

        Queue<NestedInteger> q = new LinkedList<>(nestedList);
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                NestedInteger n = q.poll();
                if (n.isInteger()) {
                    sum += depth * n.getInteger();
                } else {
                    q.addAll(n.getList());
                }
            }
            depth++;
        }
        return sum;
    }
}
```  

## 364. Nested List Weight Sum II

```java
// BFS
public int depthSumInverse(List<NestedInteger> nestedList) {
    int sum = 0;
    int res = 0;
    Deque<NestedInteger> dq = new ArrayDeque<>();
    for (NestedInteger n : nestedList) {
        dq.offerLast(n);
    }

    while (!dq.isEmpty()) {
        int size = dq.size();
        for (int i = 0; i < size; i++) {
            NestedInteger n = dq.pollFirst();
            if (n.isInteger()) {
                sum += n.getInteger(); // 上一层的结果还在sum中，再加一次的时候相当于重复加了第一层
            } else {
                dq.addAll(n.getList());
            }
        }
        res += sum;
    }
    return res;
}
```  

## 743. Network Delay Time

```java
// HashMap keep relationship between source and target and its distance
// Use PriorityQueue to get the shortest distance to get to each node
public int networkDelayTime(int[][] times, int N, int K) {
    // source -> (target -> time)
    Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
    for (int[] time : times) { // O(M)
        map.putIfAbsent(time[0], new HashMap());
        map.get(time[0]).put(time[1], time[2]);
    }

    // sort int[] in pq based on distance
    Queue<int[]> pq = new PriorityQueue<>((a,b) -> (a[0] - b[0]));

    // Add starting node [distance, to reach node]
    pq.add(new int[]{0, K});

    // check if the node has been visited
    boolean[] visited = new boolean[N + 1];

    int distance = 0;

    while(!pq.isEmpty()){
        // one node could have multiple path with multiple distance
        // pq will always give the one with shortest distance
        int[] cur = pq.remove();
        int curNode = cur[1];
        int curDist = cur[0];
        if (visited[curNode]) continue;
        visited[curNode] = true;
        distance = curDist;
        N--;
        // if curNode is the source node of an edge
        if(map.containsKey(curNode)){
            // traversal all the target node of curNode
            for(int next : map.get(curNode).keySet()){
                // update all the target node with updated distance
                pq.add(new int[]{curDist + map.get(curNode).get(next), next});
            }
        }
    }
    return N == 0 ? distance : -1;
}
```  

## 787. Cheapest Flights Within K Stops

```java
// time: O(NKLogN)
// space: O(N)
class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        // 建立邻接链表 src -> (tar -> price)
        HashMap<Integer, HashMap<Integer, Integer>> map = new HashMap<>();
        for (int[] flight : flights) {
            map.putIfAbsent(flight[0], new HashMap<>());
            map.get(flight[0]).put(flight[1], flight[2]);
        }
        Queue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        pq.offer(new int[]{0, src, K + 1});
        while (!pq.isEmpty()) {
            int[] top = pq.poll();
            int price = top[0];
            int city = top[1];
            int stop = top[2];
            if (city == dst) {
                return price;
            }
            if (stop > 0) {
                Map<Integer, Integer> tar = map.getOrDefault(city, new HashMap<>());
                for (int adj : tar.keySet()) {
                    pq.offer(new int[]{price + tar.get(adj), adj, stop - 1});
                }
            }
        }
        return -1;
    }
}
```  
