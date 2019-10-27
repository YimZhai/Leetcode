# Questions

## 1167. Minimum Cost to Connect Sticks

```java
class Solution {
    public int connectSticks(int[] sticks) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int stick : sticks) {
            pq.offer(stick);
        }
        int res = 0;
        while (pq.size() > 1) {
            int cost = pq.poll() + pq.poll();
            res += cost;
            pq.offer(cost);
        }
        return res;
    }
}
```

## 692. Top K Frequent Word

HashMap和PriorityQueue, HashMap用来记录每个词和这个词出现的次数，在insert到pq的时候，按照单词出现的次数insert  
如果次数相同，则按照字母顺序.
O(NlogN) time and O(N) space

```java
public List<String> topKFrequent(String[] words, int k) {
    List<String> res = new LinkedList<>();
    Map<String, Integer> map = new HashMap<>();
    for (String word : words) {
        map.put(word, map.getOrDefault(word, 0) + 1);
    }
    PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(
        (a, b) -> a.getValue() == b.getValue() ?
         b.getKey().compareTo(a.getKey()) : a.getValue() - b.getValue()
    );
    // insert into pq
    for (Map.Entry<String, Integer> entry : map.entrySet()) {
        pq.offer(entry);
        if (pq.size() > k) {
            pq.poll();
        }
    }
    while (!pq.isEmpty()) {
        res.add(0, pq.poll().getKey()); // pq.poll() always return the smallest.
    }
    return res;
}
```  

## 347. Top K Frequent Elements

思路同上，更换数据类型, O(NlogN) time and O(N) space

```java
class Solution {
    public List<Integer> topKFrequent(int[] nums, int k) {
        LinkedList<Integer> res = new LinkedList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        // int -> cnt
        Map<Integer, Integer> map = new HashMap<>();
        PriorityQueue<Map.Entry<Integer, Integer>> pq =
            new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            pq.offer(entry);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        while (!pq.isEmpty()) {
            res.addFirst(pq.poll().getKey());
        }
        return res;
    }
}
```  

1. Bucket Sort
建立一个数组，`List<Integer>[]`，数组下标对应的是频率，list里存储出现这么多频率的数字有哪些
O(N) time and O(N) space

```java
public List<Integer> topKFrequent(int[] nums, int k) {
    List<Integer>[] bucket = new List[nums.length + 1];
    Map<Integer, Integer> map = new HashMap<>();
    List<Integer> res = new LinkedList<>();
    for (int num : nums) {
        map.put(num, map.getOrDefault(num, 0) + 1);
    }
    // 遍历keySet()
    for (int key : map.keySet()) {
        int cnt = map.get(key);
        if (bucket[cnt] == null) {
            bucket[cnt] = new ArrayList<>();
        }
        bucket[cnt].add(key);
    }
    // 遍历bucket
    for (int i = bucket.length - 1; i >= 0 && res.size() < k; i--) {
        if (bucket[i] != null) {
            res.addAll(bucket[i]);
        }
    }
    return res;
}
```  

## 973. K Closet Point to Origin

思路：使用一个数组，记录每个点的距离，将这个数组排序，我们需要K个点，所以距离为这个数组的前K位的点就是我们的目标

```java
public int[][] kClosest(int[][] points, int K) {
    int[] power = new int[points.length];
    for (int i = 0; i < points.length; i++) {
        power[i] = cal(points[i][0], points[i][1]);
    }

    Arrays.sort(power);
    int target = power[K - 1];
    int[][] res = new int[K][2];
    int index = 0;
    for (int[] point : points) {
        if (cal(point[0], point[1]) <= target) {
            res[index] = point;
            index++;
        }
    }
    return res;
}

public int cal(int i, int j) {
    return i * i + j * j;
}
```  

2.PriorityQueue, 保持queue的size为k，在放入queue的时候，降序放入

```java
public int[][] kClosest(int[][] points, int K) {
    PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparing(a -> -a[0] * a[0] - a[1] * a[1]));
    for (int[] p : points) {
        pq.offer(p);
        if (pq.size() > K) { // poll out the farthest among the K + 1 points.
            pq.poll();
        }
    }
    int[][] ans = new int[K][2];
    while (K > 0) {
        K--;
        ans[K] = pq.poll();
    }
    return ans;
}
```  

## 215. Kth Largest Element in Array

1. Sort
2. PriorityQueue, add 默认为升序，判断queue的size和值的大小

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> pq = new PriorityQueue<>();
    for (int num : nums) {
        if (pq.size() < k || num > pq.peek()) {
            pq.add(num);
        }
        if (pq.size() > k) {
            pq.remove();
        }
    }
    return pq.peek();
}
```  
