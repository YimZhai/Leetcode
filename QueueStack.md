# Questions

## 155. Min Stack

用两个stack，第一个正常操作，第二个站只push比当前peek值小的

```java
class MinStack {

    Stack<Integer> s1;
    Stack<Integer> s2;
    /** initialize your data structure here. */
    public MinStack() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    public void push(int x) {
        s1.push(x);
        if (s2.empty() || x <= s2.peek()) {
            s2.push(x);
        }
    }

    public void pop() {
        int x = s1.pop();
        if (x == s2.peek()) {
            s2.pop();
        }
    }

    public int top() {
        return s1.peek();
    }

    public int getMin() {
        return s2.peek();
    }
}
```  

## BFS

## DFS

### 79. Word Search

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

### 212. Word Search II

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
