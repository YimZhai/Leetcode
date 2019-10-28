# Tree

## Binary Tree

### 100. Same Tree

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

```java
// non-recursion solution
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        Queue<TreeNode> q1 = new LinkedList<>();
        Queue<TreeNode> q2 = new LinkedList<>();
        q1.offer(p);
        q2.offer(q);
        while (!q1.isEmpty() || !q2.isEmpty()) {
            TreeNode np = q1.poll();
            TreeNode nq = q2.poll();
            if (np == null && nq == null) {
                continue;
            }
            if (np == null || nq == null || np.val != nq.val) {
                return false;
            }
            q1.offer(np.left);
            q1.offer(np.right);
            q2.offer(nq.left);
            q2.offer(nq.right);
        }
        if (!q1.isEmpty() || !q2.isEmpty()) {
            return false;
        }
        return true;
    }
}
```

### 572. Subtree of Another Tree

```java
// 思路同上，在初始函数里分别调用节点，对比s和t是否相同
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if (s == null) {
            return false;
        }
        if (helper(s, t)) {
            return true;
        }
        return isSubtree(s.left, t) || isSubtree(s.right, t);
    }

    public boolean helper(TreeNode s, TreeNode t) {
        if (s == null && t == null) {
            return true;
        }
        if (s == null || t == null || s.val != t.val) {
            return false;
        }
        return helper(s.left, t.left) && helper(s.right, t.right);
    }
}
```

### 297. Serialize and Deserialize Binary Tree

```java
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) {
            return "{}";
        }

        List<TreeNode> list = new ArrayList<>();
        list.add(root);
        for (int i = 0; i < list.size(); i++) {
            TreeNode node = list.get(i);
            if (node == null) {
                continue;
            }
            list.add(node.left);
            list.add(node.right);
        } // 1, 2, 3, #, #, 4, 5, #, #, #, #, #, #, #, #,

        // remove the rest "null"
        while (list.get(list.size() - 1) == null) {
            list.remove(list.size() - 1);
        }
        // 生成string
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append(list.get(0).val);
        for (int i = 1; i < list.size(); i++) {
            if (list.get(i) == null) {
                sb.append(",#");
            } else {
                sb.append(",");
                sb.append(list.get(i).val);
            }
        }
        sb.append("}");
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("{}")) {
            return null;
        }

        String[] vals = data.substring(1, data.length() - 1).split(",");
        ArrayList<TreeNode> nodes = new ArrayList<>(); // list存储已经更新过的节点
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        nodes.add(root);

        int index = 0;
        boolean isLeftChild = true; // 初始化为左子树

        for (int i = 1; i < vals.length; i++) {
            if (!vals[i].equals("#")) { // 不为空
                TreeNode node = new TreeNode(Integer.parseInt(vals[i]));
                if (isLeftChild) {
                    nodes.get(index).left = node;
                } else {
                    nodes.get(index).right = node;
                }
                nodes.add(node);
            }
            if (!isLeftChild) { // 右子树更新过后，更新 需要被更新子树 的节点
                index++;
            }
            isLeftChild = !isLeftChild; // 每更新一个值，将flag调转
        }
        return root;
    }
}
```  

### 652. Find Duplicate Subtrees

使用DFS, 遍历每一个节点及其子树，将树的结构用String表示，存在map中

```java
class Solution {
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Map<String, Integer> map = new HashMap<>();
        helper(root, res, map);
        return res;
    }

    public String helper(TreeNode root, List<TreeNode> res, Map<String, Integer> map) {
        String left = "";
        if (root.left != null) {
            left = helper(root.left, res, map);
        }
        String right = "";
        if (root.right != null) {
            right = helper(root.right, res, map);
        }
        String str = Integer.toString(root.val) + "," + left + "," + right;
        if (map.containsKey(str) && map.get(str) == 1) {
            res.add(root);
        }
        map.put(str, map.getOrDefault(str, 0) + 1);
        return str;
    }
}
```

### 103. Binary Tree Zigzag Level Order Traversal

在树的层级遍历基础上，添加一个方向变量，每遍历完一层调转方向

```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> lists = new ArrayList<>();
    if (root == null) {
        return lists;
    }
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    boolean direct = true;
    while (!q.isEmpty()) {
        int len = q.size();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            TreeNode node = q.poll();
            if (direct) {
                list.add(node.val);
            } else {
                list.add(0, node.val);
            }

            if (node.left != null) {
                q.offer(node.left);
            }
            if (node.right != null) {
                q.offer(node.right);
            }
        }
        direct = !direct;
        lists.add(list);
    }
    return lists;
}
```  

### 199. Binary Tree Right Side View

BFS,树的层级遍历，每层开始遍历的时候从最右边开始，同时将每一层遍历到的第一个节点加入结果

```java
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) {
        return res;
    }

    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    while (!q.isEmpty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            TreeNode node = q.poll();
            if (i == 0) {
                res.add(node.val);
            }
            if (node.right != null) {
                q.offer(node.right);
            }
            if (node.left != null) {
                q.offer(node.left);
            }
        }
    }
    return res;
}
```

### 543. Diameter of Binary Tree

```java
// DFS, 分别遍历左右子树，遍历的同时更新res
class Solution {
    int res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return res;
    }

    public int dfs(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = 0;
        if (node.left != null) {
            left = dfs(node.left);
        }
        int right = 0;
        if (node.right != null) {
            right = dfs(node.right);
        }
        res = Math.max(res, left + right);
        return Math.max(left, right) + 1;
    }
}
```

### 104. Maximum Depth of Binary Tree

```java
// Top Down
class Solution {
    public int res = 0;
    public int maxDepth(TreeNode root) {
        if (root == null) return res;
        helper(root, res);
        return res + 1;
    }
    private void helper(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            res = Math.max(res, depth);
        }
        helper(root.left, depth + 1);
        helper(root.right, depth + 1);
    }
}```  
```java
// Bottom Up
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        int leftDepth = maxDepth(root.left);
        int rightDepth = maxDepth(root.right);
        return Math.max(leftDepth, rightDepth) + 1;
    }
}
```  

### 112. Path Sum

```java
public boolean hasPathSum(TreeNode root, int sum) {
    if (root == null) return false; // recursion exit;

    sum -= root.val; // minus the current value
    if ((root.left == null) && (root.right == null)) { // when it is leaf node, check whether the sum equal to 0
        return sum == 0;
    }
    // check left and right
    return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
}
```

### 113. Path Sum II

```java
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> paths = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(root, sum, paths, path);
        return paths;
    }

    public void dfs(TreeNode root, int sum, List<List<Integer>> paths, List<Integer> path) {
        if (root == null) {
            return;
        }
        path.add(root.val);
        // end of one path
        if (root.left == null && root.right == null) {
            if (root.val == sum) {
                paths.add(new ArrayList(path));
            }
            return;
        }
        if (root.left != null) {
            dfs(root.left, sum - root.val, paths, path);
            path.remove(path.size() - 1); // backtracking
        }
        if (root.right != null) {
            dfs(root.right, sum - root.val, paths, path);
            path.remove(path.size() - 1);
        }
    }
}
```  

### 101. Symmetric Tree

Time and space complexity: O(N)

```java
// recursive
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return helper(root, root);
    }

    public boolean helper(TreeNode n1, TreeNode n2) {
        if (n1 == null && n2 == null) return true;
        if (n1 == null || n2 == null) return false;
        return n1.val == n2.val && helper(n1.left, n2.right) && helper(n1.right, n2.left);
    }
}
```

```java
// iterative
class Solution {
    public boolean isSymmetric(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        stack.push(root);
        while (!stack.empty()) {
            TreeNode n1 = stack.pop();
            TreeNode n2 = stack.pop();
            if (n1 == null && n2 == null) continue;
            if (n1 == null || n2 == null || n1.val != n2.val) return false;
            stack.push(n1.left);
            stack.push(n2.right);
            stack.push(n1.right);
            stack.push(n2.left);
        }
        return true;
    }
}
```

### 250. Count Univalue Subtrees

思路是先序遍历树的所有的节点，然后对每一个节点调用判断以当前节点为根的子树的所有节点是否相同

```java
/**
 * No instance variable
 * Top down
 */
class Solution {
    public int countUnivalSubtrees(TreeNode root) {
        if (root == null) return 0;
        int count = isUnival(root) ? 1 : 0;
        return count + countUnivalSubtrees(root.left) + countUnivalSubtrees(root.right);
    }

    public boolean isUnival(TreeNode node) {
        boolean bool = true;
        if (node.left != null) {
            bool = bool && (node.val == node.left.val);
            bool = bool && isUnival(node.left);
        }
        if (node.right != null) {
            bool = bool && (node.val == node.right.val);
            bool = bool && isUnival(node.right);
        }
        return bool;
    }
}

// with instance variable
// bottom up
class Solution {
    int count;
    public int countUnivalSubtrees(TreeNode root) {
        count = 0;
        helper(root);
        return count;
    }

    boolean helper(TreeNode root) {
        if (root == null) return true;
        boolean left = helper(root.left);
        boolean right = helper(root.right);
        // reach leaf node
        if (left && right &&
           (root.left == null || root.val == root.left.val) &&
           (root.right == null || root.val == root.right.val)) {
            count++;
            return true;
        }
        return false;
    }
}
```

### 105. Construct Binary Tree from Preorder and Inorder Traversal

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int inLen = inorder.length;
        int preLen = preorder.length;
        if (inLen == 0 || preLen == 0 || inLen != preLen) return null;
        // node value -> inorder index
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inLen; i++) {
            map.put(inorder[i], i);
        }
        return helper(inorder, 0, inLen - 1, preorder, 0, preLen - 1, map);
    }

    public TreeNode helper(int[] inorder, int inStart, int inEnd,
                           int[] preorder, int preStart, int preEnd, Map<Integer, Integer> map) {
        // recursion exit
        if (inStart > inEnd || preStart > preEnd) return null;
        TreeNode root = new TreeNode(preorder[preStart]);
        int rootIndex = map.get(root.val);
        // find left child and right child seperately
        TreeNode leftChild = helper(inorder, inStart, rootIndex - 1,
                                    preorder, preStart + 1, rootIndex - inStart + preStart, map);
        TreeNode rightChild = helper(inorder, rootIndex + 1, inEnd,
                                     preorder, rootIndex - inStart + preStart + 1, preEnd, map);
        root.left = leftChild;
        root.right = rightChild;
        return root;
    }
}
```

### 106. Construct Binary Tree from Inorder and Postorder Traversal

首先可以确定root为postorder的最后一位，在inorder中可以根据root将tree分为左右两部分，然后通过递归的形式将tree补全

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int inLen = inorder.length;
        int poLen = postorder.length;
        if (inLen == 0 || poLen == 0 || inLen != poLen) return null;
        // node value -> inorder index
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inLen; i++) {
            map.put(inorder[i], i);
        }
        return helper(inorder, 0, inLen - 1, postorder, 0, poLen - 1, map);
    }

    public TreeNode helper(int[] inorder, int inStart, int inEnd,
                           int[] postorder, int poStart, int poEnd, Map<Integer, Integer> map) {
        // recursion exit
        if (inStart > inEnd || poStart > poEnd) return null;
        TreeNode root = new TreeNode(postorder[poEnd]);
        int rootIndex = map.get(root.val);
        // find left child and right child seperately
        TreeNode leftChild = helper(inorder, inStart, rootIndex - 1,
                                    postorder, poStart, rootIndex - inStart + poStart - 1, map);
        TreeNode rightChild = helper(inorder, rootIndex + 1, inEnd,
                                     postorder, rootIndex - inStart + poStart, poEnd - 1, map);
        root.left = leftChild;
        root.right = rightChild;
        return root;
    }
}
```

### 257. Binary Tree Paths

```java
// Recursion
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        helper(root, res, sb);
        return res;
    }

    private void helper(TreeNode root, List<String> res, StringBuilder sb) {
        if (root == null) {
            return;
        }
        int len = sb.length();
        if (root.left == null && root.right == null) {
            sb.append(root.val);
            res.add(sb.toString());
            sb.delete(len, sb.length()); // remove last added node
            return;
        }
        sb.append(root.val + "->");
        helper(root.left, res, sb);
        helper(root.right, res, sb);
        sb.delete(len, sb.length());
        return;
    }
}
```

### 116. Populating Next Right Pointers in Each Node

首先外层循环遍历每层最左端的点，node每到一层的时候，更新当前层下一层的next指针

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Node node = root;
        while (node != null) { // traversal through level
            Node cur = node;
            while (cur != null) { // same level traversal
                if (cur.left != null) {
                    cur.left.next = cur.right;
                }
                if (cur.right != null && cur.next != null) {
                    cur.right.next = cur.next.left;
                }
                cur = cur.next;
            }
            node = node.left;
        }
        return root;
    }
}
```

### 117. Populating Next Right Pointers in Each Node II

用三个指针分别记录下一层的开始节点，每一层开始的dummy节点和遍历的指针

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Node head = null; // keeps the start of the next level
        Node prev = null; // keeps the start of each level
        Node cur = root;
        while (cur != null) { // different level
            while (cur != null) { // same level
                // each level, head only update once
                if (cur.left != null) {
                    if (prev != null) {
                        prev.next = cur.left;
                    } else {
                        head = cur.left;
                    }
                    prev = cur.left; // update prev no matter what
                }
                if (cur.right != null) {
                    if (prev != null) {
                        prev.next = cur.right;
                    } else {
                        head = cur.right;
                    }
                    prev = cur.right;
                }
                cur = cur.next;
            }
            cur = head;
            head = null;
            prev = null;
        }
        return root;
    }
}
```  

### 236. Lowest Common Ancestor of a Binary Tree

1）若p和q要么分别位于左右子树中，那么对左右子结点调用递归函数，会分别返回p和q结点的位置，而当前结点正好就是p和q的最小共同父结点，直接返回当前结点即可，这就是题目中的例子1的情况。

2）若p和q同时位于左子树，这里有两种情况，一种情况是left会返回p和q中较高的那个位置，而right会返回空，所以我们最终返回非空的left即可，这就是题目中的例子2的情况。还有一种情况是会返回p和q的最小父结点，就是说当前结点的左子树中的某个结点才是p和q的最小父结点，会被返回。

3）若p和q同时位于右子树，同样这里有两种情况，一种情况是right会返回p和q中较高的那个位置，而left会返回空，所以我们最终返回非空的right即可，还有一种情况是会返回p和q的最小父结点，就是说当前结点的右子树中的某个结点才是p和q的最小父结点，会被返回。

[原文链接](https://blog.csdn.net/qq_43322057/article/details/84786676)

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q); // 从左子树里查找p和q
        TreeNode right = lowestCommonAncestor(root.right, p, q); // 从右子树里查找p和q
        if (left != null && right != null) { // p和q分别在左右子树中
            return root; // 当前节点即为LCA
        }
        // 节点在左子树中
        if (left != null) {
            return left;
        } else {
            // 节点在右子树
            return right;
        }
    }
}
```

***

## Binary Search Tree

### 449. Serialize and Deserialize BST

Preorder traversal + queue

```java
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if (root == null) return "null";
        //traverse it recursively if you want to, I am doing it iteratively here
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while (!st.empty()) {
            root = st.pop();
            sb.append(root.val).append(",");
            if (root.right != null) st.push(root.right);
            if (root.left != null) st.push(root.left);
        }
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.equals("null")) return null;
        String[] strs = data.split(",");
        Queue<Integer> q = new LinkedList<>();
        for (String e : strs) {
            q.offer(Integer.parseInt(e));
        }
        return getNode(q);
    }

    // some notes:
    //   5
    //  3 6
    // 2   7
    private TreeNode getNode(Queue<Integer> q) { //q: 5,3,2,6,7
        if (q.isEmpty()) return null;
        TreeNode root = new TreeNode(q.poll());//root (5)
        Queue<Integer> samllerQueue = new LinkedList<>();
        while (!q.isEmpty() && q.peek() < root.val) {
            samllerQueue.offer(q.poll());
        }
        //smallerQueue : 3,2   storing elements smaller than 5 (root)
        root.left = getNode(samllerQueue);
        //q: 6,7   storing elements bigger than 5 (root)
        root.right = getNode(q);
        return root;
    }
}
```  

### 98. Validate Binary Search Tree

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        // use long in case of [2147483647]
        return dfs(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean dfs(TreeNode node, long low, long high) {
        if (node == null) {
            return true;
        }
        if (node.val <= low || node.val >= high) return false;
        // 判断左节点将high设为根节点，判断右节点将low设为根节点
        return dfs(node.left, low, node.val) && dfs(node.right, node.val, high);
    }
}
```  

## 96. Unique Binary Search Trees

```java
// dp solution, calculate Catalan Number
// dp[2] = dp[0] * dp[1] + dp[1] * dp[0]
// dp[3] = dp[0] * dp[2] + dp[1] * dp[1] + dp[2] * dp[0]
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[i - 1 - j];
            }
        }
        return dp[n];
    }
}
```

## 95. Unique Binary Search Trees II

```java
// 思路：从1到n-1中选取作为root的点，将两边点分别construct
class Solution {
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new ArrayList<TreeNode>();
        }
        return helper(1, n);
    }

    public List<TreeNode> helper(int start, int end) {
        List<TreeNode> list = new ArrayList<>();
        if (start > end) { // recursion exit
            list.add(null);
            return list;
        }
        for (int i = start; i <= end; i++) {
            // all the possible left subtree
            List<TreeNode> left = helper(start, i - 1);
            // all the possible right subtree
            List<TreeNode> right = helper(i + 1, end);
            // connect left tree and right tree to the root i
            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = l;
                    root.right = r;
                    list.add(root);
                }
            }
        }
        return list;
    }
}
```

## 426. Convert Binary Search Tree to Sorted Doubly Linked List

```java
// 思路，用中序遍历，第一步更新头尾之间的指针，第二步更新头尾互指
class Solution {
    Node pre = null;
    public Node treeToDoublyList(Node root) {
        if (root == null) {
            return null;
        }
        Node head = new Node(0, null, null);
        pre = head;
        inorder(root); // 建立链接
        // 建立头尾的链接
        pre.right = head.right;
        head.right.left = pre;
        return head.right;
    }

    public void inorder(Node node) {
        if (node == null) {
            return;
        }
        inorder(node.left);
        pre.right = node; // 建立双向指针
        node.left = pre;
        pre = node; // 更新pre
        inorder(node.right);
    }
}

***

## N-ary Tree

## 428. Serialize and Deserialize N-ary Tree

```java
// 使用递归的方法解决
// 时间复杂度 O(N)
// 同样的方法也可以解决二叉树的序列化，只需要跳过添加children size的步骤即可
class Codec {

    String NULL_NODE = "#";
    String SPLITER = ",";
    // Encodes a tree to a single string.
    public String serialize(Node root) {
        StringBuilder sb = new StringBuilder();
        serializeHelper(root, sb);
        return sb.toString();
    }

    public void serializeHelper(Node node, StringBuilder sb) {
        if (node == null) {
            sb.append(NULL_NODE);
            sb.append(SPLITER);
        } else {
            sb.append(node.val);
            sb.append(SPLITER);
            sb.append(node.children.size());
            sb.append(SPLITER);
            for (Node n : node.children) {
                serializeHelper(n, sb);
            }
        }
    }

    // Decodes your encoded data to tree.
    public Node deserialize(String data) {
        Deque<String> dq = new LinkedList(Arrays.asList(data.split(SPLITER)));
        return deserializeHelper(dq);
    }

    public Node deserializeHelper(Deque<String> dq) {
        String str = dq.removeFirst();
        if (str.equals(NULL_NODE)) {
            return null;
        }
        int val = Integer.valueOf(str); // 该节点的值
        int childrenNum = Integer.valueOf(dq.removeFirst()); // 该节点孩子节点的个数
        Node root = new Node(val, new ArrayList<Node>());
        for (int i = 0; i < childrenNum; i++) {
            root.children.add(deserializeHelper(dq));
        }
        return root;
    }
}
```  

***

## Trie Tree
