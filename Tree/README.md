# Binary Tree

## Basic

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
        ArrayList<TreeNode> nodes = new ArrayList<>();
        TreeNode root = new TreeNode(Integer.parseInt(vals[0]));
        nodes.add(root);

        int index = 0;
        boolean isLeftChild = true;

        for (int i = 1; i < vals.length; i++) {
            if (!vals[i].equals("#")) {
                TreeNode node = new TreeNode(Integer.parseInt(vals[i]));
                if (isLeftChild) {
                    nodes.get(index).left = node;
                } else {
                    nodes.get(index).right = node;
                }

                nodes.add(node);
            }
            if (!isLeftChild) {
                index++;
            }
            isLeftChild = !isLeftChild;
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


# N-ary Tree

## Trie Tree