# Tree

## Binary Tree

### Three Way Traversal

```java
// Preorder, Recursion
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        helper(root, res);
        return res;
    }

    private void helper(TreeNode node, List<Integer> res) {
        if (node == null) return;
        res.add(node.val);
        helper(node.left, res);
        helper(node.right, res);
    }
}

// Iteration
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.empty()) {
            TreeNode node = stack.pop();
            res.add(node.val);
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return res;
    }
}
```

```java
// Inorder, recursion
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        helper(root, res);
        return res;
    }

    private void helper(TreeNode node, List<Integer> res) {
        if (node == null) return;
        helper(node.left, res);
        res.add(node.val);
        helper(node.right, res);
    }
}

// Iteration
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.empty()) {
            while (cur != null) { // find the most left
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            res.add(cur.val);
            cur = cur.right;
        }
        return res;
    }
}
```  

```java
// Postorder, recursion
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        helper(root, res);
        return res;
    }

    private void helper(TreeNode node, List<Integer> res) {
        if (node == null) return;
        helper(node.left, res);
        helper(node.right, res);
        res.add(node.val);
    }
}

// Iteration
/**
 * 1. Reverse preorder traversal result
 * Result is correct, however, the traversal order is not correct
 */
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        LinkedList<Integer> res = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        if (root == null) return res;

        stack.push(root);
        while (!stack.empty()) {
            TreeNode node = stack.pop();
            res.addFirst(node.val);
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
        }
        return res;
    }
}

// correct traversal order
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        LinkedList<Integer> res = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();

        while (!stack.empty() || root != null) {
            // find leaf nodes
            while (root != null) {
                stack.push(root);
                if (root.left != null) {
                    root = root.left;
                } else {
                    root = root.right;
                }
            }
            TreeNode node = stack.pop();
            res.add(node.val);
            if (!stack.empty() && stack.peek().left == node) { // current node is left child
                root = stack.peek().right;
            }
        }
        return res;
    }
}
```  

### 110. Balanced Binary Tree

```java
// bottom up
class Solution {
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = getDepth(root.left);
        int right = getDepth(root.right);
        return Math.abs(left - right) <= 1 && isBalanced(root.left) && isBalanced(root.right);
    }

    public int getDepth(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return Math.max(getDepth(node.left), getDepth(node.right)) + 1;
    }
}
```  

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
    if ((root.left == null) && (root.right == null)) {
        // when it is leaf node, check whether the sum equal to 0
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
// Recursion backtracking
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

### 235. Lowest Common Ancestor of a Binary Search Tree

```java
// BT的解法也适用
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val) { // p，q都在左子树
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) { // p, q都在右子树
            return lowestCommonAncestor(root.right, p, q);
        } else { // 分别在左右子树，直接返回root
            return root;
        }
    }
}
```  

### 1130. Minimum Cost Tree From Leaf Values

```java
// 在建树的过程中，每比较一次(a,b)的最小值，都要花费a*b，所以问题转化为了移除数组元素直到剩余一个的最小花费
// 当a <= b，我们要移除a，同时我要让b尽可能小
class Solution {
    public int mctFromLeafValues(int[] arr) {
        Stack<Integer> stack = new Stack<>();
        stack.push(Integer.MAX_VALUE);
        int res = 0;
        for (int a : arr) {
            while (a >= stack.peek()) {
                int mid = stack.pop();
                res += mid * Math.min(stack.peek(), a);
            }
            stack.push(a);
        }
        while (stack.size() > 2) {
            res += stack.pop() * stack.peek();
        }
        return res;
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

### 285. Inorder Successor in BST

```java
// 第一个思路，中序遍历得到整个树的结果，再用二分查找，找到目标值的下一个值
// 空间复杂度为O(N)，时间为中序遍历O(N) + 二分查找O(logN)
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
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        Stack<TreeNode> stack = new Stack<>();
        List<TreeNode> list = new ArrayList<>();
        TreeNode runner = root;
        while (runner != null || !stack.empty()) {
            while (runner != null) {
                stack.push(runner);
                runner = runner.left;
            }
            runner = stack.pop();
            list.add(runner);
            runner = runner.right;
        }
        int left = 0;
        int right = list.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (list.get(mid).val == p.val) {
                if (mid != list.size() - 1) {
                    return list.get(mid + 1);
                } else {
                    return null;
                }
            } else if (list.get(mid).val > p.val) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return null;
    }
}
```  

```java
// O(H) solution, worst case could be O(N)
class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode succ = null;
        while (root != null) {
            if (p.val < root.val) {
                succ = root;
                root = root.left;
            } else {
                root = root.right;
            }
        }
        return succ;
    }
}
```  

### 173. Binary Search Tree Iterator

```java
// 考点，BST的非递归中序遍历
class BSTIterator {

    Stack<TreeNode> stack;
    public BSTIterator(TreeNode root) {
        stack = new Stack<>();
        while (root != null) { // 将左子树全部放入stack
            stack.push(root);
            root = root.left;
        }
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode node = stack.pop();
        int res = node.val;
        // 若右子树不为空，将右子树的全部左子树放入stack
        if (node.right != null) {
            node = node.right;
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
        }
        return res;
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.empty();
    }
}
```  

### 108. Convert Sorted Array to Binary Search Tree

```java
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        TreeNode root = helper(nums, 0, nums.length - 1);
        return root;
    }

    public TreeNode helper(int[] nums, int lo, int hi) {
        if (lo > hi) {
            return null;
        }
        int mid = (lo + hi) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = helper(nums, lo, mid - 1);
        node.right = helper(nums, mid + 1, hi);
        return node;
    }
}
```  

### 700. Search in a Binary Search Tree

```java
// O(H) time
// recursion
class Solution {
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) {
            return null;
        }
        if (root.val == val) {
            return root;
        }
        if (val < root.val) {
            return searchBST(root.left, val);
        }
        return searchBST(root.right, val);
    }
}

// iteration
class Solution {
    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null) {
            return null;
        }
        while (root != null) {
            if (root.val == val) {
                return root;
            } else if (root.val > val) {
                root = root.left;
            } else {
                root = root.right;
            }
        }
        return null;
    }
}
```  

### 701. Insert into a Binary Search Tree

```java
// recursion, O(H) time and space, O(logN) best case, O(N) worst case
class Solution {
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return new TreeNode(val);
        }
        if (val < root.val) {
            root.left = insertIntoBST(root.left, val);
        } else {
            root.right = insertIntoBST(root.right, val);
        }
        return root;
    }
}

// iteration
class Solution {
    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            return root;
        }
        TreeNode runner = root;
        while (runner != null) {
            if (val < runner.val) {
                if (runner.left != null) {
                    runner = runner.left;
                } else {
                    runner.left = new TreeNode(val);
                    break;
                }
            } else {
                if (runner.right != null) {
                    runner = runner.right;
                } else {
                    runner.right = new TreeNode(val);
                    break;
                }
            }
        }
        return root;
    }
}
```  

### 450. Deletion in a BST

```java
// 3种情况
// 1.节点为叶子节点，直接删除
// 2.节点只有一个孩子节点，替换
// 3.节点下面有多个节点，找到最近的successor或者precessor，替换
// O(H) time and space, best case O(logN), worst case O(N)
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (key < root.val) {
            root.left = deleteNode(root.left, key);
        } else if (key > root.val) {
            root.right = deleteNode(root.right, key);
        } else { // 找到需要删除的节点
            // 处理没有子节点和只有一个子节点的情况
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            }
            // 处理两个子节点
            TreeNode succ = root.right;
            while (succ.left != null) {
                succ = succ.left;
            }
            root.val = succ.val; // 交换节点的值，此时再次递归调用函数处理右子树，删除succ
            root.right = deleteNode(root.right, succ.val);
        }
        return root;
    }
}

// iteration O(1) space
class Solution {
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null || root.val == key) {
            return deleteRoot(root);
        }
        TreeNode runner = root;
        while (true) {
            if (runner.val > key) {
                if (runner.left == null || runner.left.val == key) {
                    runner.left = deleteRoot(runner.left);
                    break;
                }
                runner = runner.left;
            } else {
                if (runner.right == null || runner.right.val == key) {
                    runner.right = deleteRoot(runner.right);
                    break;
                }
                runner = runner.right;
            }
        }
        return root;
    }

    public TreeNode deleteRoot(TreeNode node) {
        if (node == null) {
            return null;
        }
        if (node.right == null) {
            return node.left;
        }
        TreeNode cur = node.right;
        while (cur.left != null) {
            cur = cur.left;
        }
        // 将node的左子树接到node的successor的左子树的位置
        cur.left = node.left;
        return node.right;
    }
}
```  

### 703. Kth Largest Element in a Stream

```java
// 使用最小堆，将size维护在k，每次从堆里poll出来的都是最小的元素
// 时间复杂度 O(KLogN)，堆的插入和删除操作都是LogN的时间
class KthLargest {

    private PriorityQueue<Integer> pq;
    private int kth;
    public KthLargest(int k, int[] nums) {
        kth = k;
        pq = new PriorityQueue<>((a, b) -> a - b); // min heap
        for (int i = 0; i < nums.length; i++) {
            pq.offer(nums[i]);
            if (pq.size() > kth) {
                pq.poll();
            }
        }
    }

    public int add(int val) {
        pq.offer(val);
        if (pq.size() > kth) {
            pq.poll();
        }
        return pq.peek();
    }
}
```  

### 96. Unique Binary Search Trees

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

### 95. Unique Binary Search Trees II

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

### 426. Convert Binary Search Tree to Sorted Doubly Linked List

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
```  

### 220. Contains Duplicate III

```java
// 建立一个TreeSet，保持size为k, 时间复杂度 O(NLogK)
class Solution {
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> set = new TreeSet<>();
        int i = 0;
        while (i < nums.length) {
            Long floor = set.floor((long) nums[i]); // 小于等于nums[i]的最大值
            Long ceiling = set.ceiling((long) nums[i]); // 大于等于nums[i]的最小值
            if ((floor != null && nums[i] - floor <= t)
               || (ceiling != null && ceiling - nums[i] <= t)) {
                return true;
            }
            set.add((long) nums[i]);
            i++;
            if (set.size() > k) {
                set.remove((long) nums[i - 1 - k]);
            }
        }
        return false;
    }
}
```  

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

### 208. Implement Trie (Prefix Tree)

```java
class TrieNode {
    boolean isEndOfWord;
    TrieNode[] children;

    public TrieNode() {
        isEndOfWord = false;
        children = new TrieNode[26];
    }
}

class Trie {

    private TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        TrieNode runner = root;
        for (char c : word.toCharArray()) {
            if (runner.children[c - 'a'] == null) {
                runner.children[c - 'a'] = new TrieNode();
            }
            runner = runner.children[c - 'a'];
        }
        runner.isEndOfWord = true;
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode runner = root;
        for (char c : word.toCharArray()) {
            if (runner.children[c - 'a'] == null) {
                return false;
            }
            runner = runner.children[c - 'a'];
        }
        // insert(apple), search(app), app is done, but app.isEndOfWord = false
        return runner.isEndOfWord;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode runner = root;
        for (char c : prefix.toCharArray()) {
            if (runner.children[c - 'a'] == null) {
                return false;
            }
            runner = runner.children[c - 'a'];
        }
        return true;
    }
}
```  

### 677. Map Sum Pairs

```java
class TrieNode {
    int val;
    TrieNode[] children;
    public TrieNode(int val) {
        this.val = val;
        children = new TrieNode[256];
    }

    public void update(int val) {
        this.val += val;
    }

    public void updateDuplicate(int oldVal, int newVal) {
        this.val += newVal - oldVal;
    }
}
class MapSum {

    TrieNode root;
    HashMap<String, Integer> map;
    /** Initialize your data structure here. */
    public MapSum() {
        root = new TrieNode(0);
        map = new HashMap<>();
    }

    public void insert(String key, int val) {
        TrieNode runner = root;
        if (map.containsKey(key)) {
            for (char c : key.toCharArray()) {
                runner = runner.children[c];
                runner.updateDuplicate(map.get(key), val);
            }
        } else {
            for (char c : key.toCharArray()) {
                if (runner.children[c] == null) {
                    runner.children[c] = new TrieNode(0);
                }
                runner = runner.children[c];
                runner.update(val);
            }
            map.put(key, val);
        }
    }

    public int sum(String prefix) {
        int res = 0;
        TrieNode runner = root;
        for (char c : prefix.toCharArray()) {
            if (runner.children[c] != null) {
                runner = runner.children[c];
                res = runner.val;
            } else {
                return 0;
            }
        }
        return res;
    }
}
```  

### 648. Replace Words

```java
class TrieNode {
    char val;
    TrieNode[] children;
    boolean isEnd;

    public TrieNode(char val) {
        this.val = val;
        children = new TrieNode[26];
        isEnd = false;
    }
}
class Solution {
    public String replaceWords(List<String> dict, String sentence) {
        String[] words = sentence.split(" ");
        TrieNode root = buildTrie(dict);
        StringBuilder sb = new StringBuilder();
        for (String word : words) {
            sb.append(replace(root, word));
            sb.append(" ");
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    public String replace(TrieNode root, String word) {
        StringBuilder sb = new StringBuilder();
        TrieNode runner = root;
        for (char c : word.toCharArray()) {
            if (runner.children[c - 'a'] == null) {
                return word;
            } else {
                runner = runner.children[c - 'a'];
                sb.append(runner.val);
                if (runner.isEnd) {
                    break;
                }
            }
        }
        return sb.toString();
    }

    public TrieNode buildTrie(List<String> dicts) {
        TrieNode root = new TrieNode(' ');
        for (String dict : dicts) {
            TrieNode runner = root;
            for (char c : dict.toCharArray()) {
                if (runner.children[c - 'a'] == null) {
                    runner.children[c - 'a'] = new TrieNode(c);
                }
                runner = runner.children[c - 'a'];
            }
            runner.isEnd = true;
        }
        return root;
    }
}
```  

### 421. Maximum XOR of Two Numbers in an Array

```java
class TrieNode {
    TrieNode zero;
    TrieNode one;
}

class Solution {
    public int findMaximumXOR(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        // 建树
        TrieNode root = new TrieNode();
        for (int num : nums) {
            TrieNode runner = root;
            for (int i = 31; i >= 0; i--) {
                int bit = (num >> i) & 1;
                if (bit == 1) {
                    if (runner.one == null) {
                        runner.one = new TrieNode();
                    }
                    runner = runner.one;
                } else {
                    if (runner.zero == null) {
                        runner.zero = new TrieNode();
                    }
                    runner = runner.zero;
                }
            }
        }

        int res = Integer.MIN_VALUE;
        for (int num : nums) {
            TrieNode runner = root;
            int xor = 0;
            for (int i = 31; i >= 0; i--) {
                int bit = (num >> i) & 1;
                if (bit == 1) {
                    // 当前位为1，如果0的子树位置不为空，说明当前i位置可以取xor
                    // 更新临时变量xor的值
                    if (runner.zero != null) {
                        runner = runner.zero;
                        xor += 1 << i;
                    } else {
                        runner = runner.one;
                    }
                } else {
                    // 此处同理
                    if (runner.one != null) {
                        runner = runner.one;
                        xor += 1 << i;
                    } else {
                        runner = runner.zero;
                    }
                }
                res = Math.max(res, xor);
            }
        }
        return res;
    }
}
```  

### 642. Design Search Autocomplete System

```java
class TrieNode {
    TrieNode[] children;
    Map<String, Integer> sens;

    public TrieNode() {
        children = new TrieNode[128];
        sens = new HashMap<>();
    }
}
class AutocompleteSystem {

    TrieNode root;
    String numSign; // 记录所有输入
    public AutocompleteSystem(String[] sentences, int[] times) {
        root = new TrieNode();
        numSign = "";
        for (int i = 0; i < times.length; i++) {
            buildTrie(sentences[i], times[i]);
        }
    }

    public void buildTrie(String sentence, int times) {
        TrieNode runner = root;
        for (char c : sentence.toCharArray()) {
            if (runner.children[c] == null) {
                runner.children[c] = new TrieNode();
            }
            runner = runner.children[c];
            runner.sens.put(sentence, runner.sens.getOrDefault(sentence, 0) + times);
        }
    }

    public List<String> input(char c) {
        if (c == '#') {
            buildTrie(numSign, 1);
            numSign = "";
            return new ArrayList<>();
        }
        numSign += c;

        TrieNode runner = root;
        for (char ch : numSign.toCharArray()) {
            if (runner.children[ch] == null) {
                return new ArrayList<>();
            }
            runner = runner.children[ch];
        }
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>((a, b) -> {
            if (a.getValue() != b.getValue()) {
                return b.getValue() - a.getValue();
            }
            return a.getKey().compareTo(b.getKey());
        });
        pq.addAll(runner.sens.entrySet());
        List<String> res = new ArrayList<>();
        int k = 0;
        while (k < 3 && !pq.isEmpty()) {
            res.add(pq.poll().getKey());
            k++;
        }
        return res;
    }
}
```  

### 425. Word Squares

```java
// 思路：首先我们遍历words，每次选取一点词放入list，根据放入的词进行搜索
// 观察可以发现，放入第二个词的要求是这个次的起点必须是第一个词的第二个字符，第三个词是第一个词和第二个词的第三个字符拼接起来，以此类推
// 搜索的过程也就变成了Trie tree中的startWith
// 因为要找到所有符合情况的排列组合，想到了使用backtrack，第一层backtrack是每次加的第一个词，第二层是后面要加的词
// 复杂度：N个词，每次词的长度为L，空间复杂度：O(NL)，时间：O(NL26^L),backtrack可能会遍历到trie的所有节点，每次搜索的时间是L
class TrieNode {
    TrieNode[] children;
    List<String> lists;

    public TrieNode() {
        children = new TrieNode[26];
        lists = new ArrayList<>();
    }
}
class Solution {
    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> res = new ArrayList<>();
        TrieNode root = buildTrie(words);
        List<String> list = new ArrayList<>();
        for (String word : words) { // 先加进去一个词再开始backtrack
            list.add(word);
            backtrack(root, word, res, list);
            list.remove(list.size() - 1);
        }
        return res;
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
                runner.lists.add(word);
            }
        }
        return root;
    }

    public void backtrack(TrieNode root, String word, List<List<String>> res, List<String> list) {
        if (list.size() == word.length()) {
            res.add(new ArrayList(list));
            return;
        }
        int index = list.size();
        String str = ""; // 要查找的目标单词
        for (String s : list) {
            str += s.charAt(index);
        }
        List<String> lists = search(root, str);
        for (String s : lists) {
            list.add(s);
            backtrack(root, s, res, list);
            list.remove(list.size() - 1);
        }
    }

    public List<String> search(TrieNode root, String s) {
        List<String> res = new ArrayList<>();
        TrieNode runner = root;
        for (char c : s.toCharArray()) {
            if (runner.children[c - 'a'] == null) {
                return new ArrayList<>();
            }
            runner = runner.children[c - 'a'];
        }
        res.addAll(runner.lists);
        return res;
    }
}
```
