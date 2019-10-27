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
//
```  

## 53. N-Queens II

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
