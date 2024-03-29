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

## 218. The Skyline Problem

```java
// 使用最大堆， O(NlogN)
class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> res = new ArrayList<>();
        List<int[]> height = new ArrayList<>();
        for (int[] building : buildings) {
            // 将起始点标记为负高度
            // 终点标记为正高度
            height.add(new int[]{building[0], -building[2]});
            height.add(new int[]{building[1], building[2]});
        }
        // 将高度按照起始点排序，如果起始点相同，按照高度排序
        Collections.sort(height, (a, b) -> {
            if (a[0] == b[0]) {
               return a[1] - b[1];
            }
            return a[0] - b[0];
        });
        // max heap
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        int prev = 0;
        pq.offer(prev);
        for (int[] h : height) {
            if (h[1] < 0) { // 起始点
                pq.offer(-h[1]);
            } else { // 到达该高度的终点
                pq.remove(h[1]);
            }
            int cur = pq.peek(); // 当前最大高度
            if (cur != prev) { // 更新当前最大高度
                res.add(new ArrayList<>(Arrays.asList(h[0], cur)));
                prev = cur;
            }
        }
        return res;
    }
}
```  

## 252. Meeting Room

根据每段会议的开始时间sort数组，从第二个元素开始遍历，比较开始时间和上一个会议的结束时间，发现重合则返回false

```java
public class IntervalComparator implements Comparator<int[]> {
    @Override
    public int compare(int[] a, int[] b) {
        return a[0] - b[0];
    }
}
class Solution {
    public boolean canAttendMeetings(int[][] intervals) {
        int len = intervals.length;
        if (len <= 1) {
            return true;
        }

        Arrays.sort(intervals, new IntervalComparator());

        for (int i = 1; i < len; i++) {
            if (intervals[i][0] < intervals[i - 1][1]) {
                return false;
            }
        }
        return true;
    }
}
```

## 253. Meeting Room II

1. Priority Queue, 会议按照开始时间ascending, PQ按照结束时间ascending

```java
public int minMeetingRooms(int[][] intervals) {
    int len = intervals.length;
    if (intervals == null || len == 0) {
        return 0;
    }

    // Sort the intervals by start time
    Arrays.sort(intervals, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) { return a[0] - b[0]; }
    });

    // Use a min heap to track the minimum end time of merged intervals
    PriorityQueue<int[]> heap =
        new PriorityQueue<int[]>(intervals.length, new Comparator<int[]>() {
        public int compare(int[] a, int[] b) { return a[1] - b[1]; }
    });

    // start with the first meeting
    heap.offer(intervals[0]);

    for (int i = 1; i < len; i++) {
        // get the meeting which finish earliest.
        int[] meeting = heap.poll();

        // the meeting needs a new room
        if (intervals[i][0] < meeting[1]) {
            heap.offer(intervals[i]);
        } else {
            // no overlap, update current meeting.
            meeting[1] = intervals[i][1];
        }
        // put the meeting back
        heap.offer(meeting);
    }
    return heap.size();
}
```  

2.同样的思路，使用两个数组解决

```java
public int minMeetingRooms(int[][] intervals) {
    int len = intervals.length;
    int[] start = new int[len];
    int[] end = new int[len];
    for (int i = 0; i < len; i++) {
        start[i] = intervals[i][0];
        end[i] = intervals[i][1];
    }
    Arrays.sort(start);
    Arrays.sort(end);
    int rooms = 0;
    int endIndex = 0;
    for (int i = 0; i < len; i++) {
        if (start[i] < end[endIndex]) {
            rooms++;
        } else {
            endIndex++;
        }
    }
    return rooms;
}
```  
