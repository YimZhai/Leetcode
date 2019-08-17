## Easy
### 771. Jewels and Stones
1. Traversal two String, put Jewels in set, and check if stones has jewels.
```java
public int numJewelsInStones(String J, String S) {
    char[] jewels = J.toCharArray();
    Set<Character> set = new HashSet<>();
    for (char c : jewels) {
        set.add(c);
    }
    int cnt = 0;
    char[] stones = S.toCharArray();
    for (char s : stones) {
        if (set.contains(s)) {
            cnt++;
        }
    }
    return cnt;
}
```

### 811. Subdomain Visit Count
1. HashMap<domain, times>, 遍历cpdomains, populate map, traversal through keySet(), add to list.
```java
public List<String> subdomainVisits(String[] cpdomains) {
    List<String> list = new ArrayList<>();
    if (cpdomains.length == 0) {
        return list;
    }

    Map<String, Integer> map = new HashMap<>();
    for (String s : cpdomains) {
        int times = Integer.parseInt(s.split(" ")[0]);
        String domain = s.split(" ")[1];
        for (int i = 0; i < domain.length(); i++) {
            if (domain.charAt(i) == '.') {
                String d = domain.substring(i + 1);
                map.put(d, map.getOrDefault(d, 0) + times);
            }
        }
        map.put(domain, map.getOrDefault(domain, 0) + times);
    }

    for (String s : map.keySet()) {
        list.add(map.get(s) + " " + s);
    }
    return list;
}
```  

## Medium
### 3. Longest Substring Without Repeating Characters
1. 双指针， 一前一后，如果后不在set里，添加进去，更新后，更新len，否则，删除前，更新前
```java
public int lengthOfLongestSubstring(String s) {
    int len = 0;
    if (s == null || s.length() == 0) {
        return len;
    }

    char[] arr = s.toCharArray();
    Set<Character> set = new HashSet<>();
    int lo = 0;
    int hi = 0;
    while (hi < arr.length) {
        if (!set.contains(arr[hi])) {
            set.add(arr[hi]);
            hi++;
            len = Math.max((hi - lo), len);
        } else {
            set.remove(arr[lo]);
            lo++;
        }
    }
    return len;

}
```  
2. Optimization 使用map代替set，map存char对应的上一次出现重复的地方，免去i一个一个更新和判断
```java
public int lengthOfLongestSubstring(String s) {
    int n = s.length();
    int len = 0;
    Map<Character, Integer> map = new HashMap<>();
    for (int j = 0, i = 0; j < n; j++) {
        if (map.containsKey(s.charAt(j))) {
            i = Math.max(map.get(s.charAt(j)), i); // 直接跳到j的最新位置
        }
        len = Math.max(len, j - i + 1);
        map.put(s.charAt(j), j + 1);
    }
    return len;
}
```  

### 49. Grouped Anagrams
1. HashMap<String, List>, key是每个string sort过之后的，遍历一遍，判断当前string sort过之后是否存在在map中
```java
public List<List<String>> groupAnagrams(String[] strs) {
    if (strs.length == 0) {
        return new ArrayList();
    }

    Map<String, List> map = new HashMap<>();
    for (String str : strs) {
        char[] chs = str.toCharArray();
        Arrays.sort(chs);
        String key = String.valueOf(chs);
        if (!map.containsKey(key)) {
            map.put(key, new ArrayList<>());
        }
        map.get(key).add(str);
    }
    return new ArrayList(map.values());
}
```  

### 560. Subarray Sum Equals K
1. O(n^2), 一个数组记录叠加到当前index时，前面的数值总和，双指针一前一后，判断差值是否为k，更新cnt
```java
public int subarraySum(int[] nums, int k) {
    int cnt = 0;
    int[] sum = new int[nums.length + 1];
    sum[0] = 0;
    for (int i = 1; i <= nums.length; i++) {
        sum[i] = nums[i - 1] + sum[i - 1];
    }
    for (int lo = 0; lo < nums.length; lo++) {
        for (int hi = lo + 1; hi <= nums.length; hi++) {
            if (sum[hi] - sum[lo] == k) {
                cnt++;
            }
        }
    }
    return cnt;
}
```
2. O(n), HashMap, <curricularSum, numOfOccurance>, 如果在两个sum的区间内，则sum[j] - k = sum[i], i一定存在map的entry.
```java
public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 1);
    int cnt = 0;
    int sum = 0;
    for (int i = 0; i < nums.length; i++) {
        sum += nums[i];
        if (map.containsKey(sum - k)) {
            cnt += map.get(sum - k);
        }
        map.put(sum, map.getOrDefault(sum, 0) + 1);
    }
    return cnt;
}
```  

### 380. Insert Delete GetRandom O(1)
1. HashMap记录val和在ArrayList对应的下标，remove的时候如不是删最后一位则换位置
```java
class RandomizedSet {
    ArrayList<Integer> nums;
    HashMap<Integer, Integer> locs;
    java.util.Random rand = new java.util.Random();

    /** Initialize your data structure here. */
    public RandomizedSet() {
        nums = new ArrayList<Integer>();
        locs = new HashMap<Integer, Integer>();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        boolean contain = locs.containsKey(val);
        if (contain) return false;
        locs.put( val, nums.size());
        nums.add(val);
        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        boolean contain = locs.containsKey(val);
        if (!contain) return false;
        int loc = locs.get(val);
        if (loc < nums.size() - 1) { // not the last one, swap
            int lastone = nums.get(nums.size() - 1);
            nums.set(loc, lastone);
            locs.put(lastone, loc);
        }
        locs.remove(val);
        nums.remove(nums.size() - 1);
        return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
        return nums.get( rand.nextInt(nums.size()) );
    }
}
```  

### 609. Find Duplicate File in System
1. HashMap，括号内的作为key，value是List<String>
```java
public List<List<String>> findDuplicate(String[] paths) {
    Map<String, List<String>> map = new HashMap<>();
    for (String path : paths) {
        String[] parts = path.split(" ");
        for (int i = 1; i < parts.length; i++) {
            int k = parts[i].indexOf("(");
            String content = parts[i].substring(k, parts[i].length() - 1);
            map.putIfAbsent(content, new LinkedList<>());
            map.get(content).add(parts[0] + "/" + parts[i].substring(0, k));
        }
    }
    List<List<String>> r = new LinkedList<>();
    for (List<String> list : map.values())
        if (list.size() > 1)
            r.add(list);
    return r;
}
```  


