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
