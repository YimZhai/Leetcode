# Questions

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
