### 1. Two Sum
1. Brute Force, two level for loop.
2. One Pass Hash
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int n = target - nums[i];
        if (map.containsKey(n)) {
            return new int[] {i, map.get(n)};
        }
        map.put(nums[i], i);
    }
    return new int[] {};
}
```  
### 53. Maximum Subarray
1. 如果sum小于0， 不管下一个数是正是负，肯定会比这个数加上sum的值大
```java
public int maxSubArray(int[] nums) {
    int sum = Integer.MIN_VALUE;
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < nums.length; i++) {
        if (sum < 0) {
            sum = nums[i];
        } else {
            sum += nums[i];
        }
        max = Math.max(max, sum);
    }
    return max;
}
```  

### 121. Best time to buy and sell stock
1. 两个值，分别记录最大和最小，O(n)遍历一遍，如果price[i] < low更新low, 否则更新max
```java
public int maxProfit(int[] prices) {
    int profit = 0;
    if (prices.length < 2) {
        return profit;
    }
    int low = prices[0];
    for (int i = 0; i < prices.length; i++) {
        if (prices[i] < low) {
            low = prices[i];
        } else {
            profit = Math.max(profit, (prices[i] - low));
        }
    }
    return profit;
}
```  
### 509. Fibonacci Number
1. 使用一个数组，计算出每个下标对应的值，返回array[N].
```java
public int fib(int N) {
    if (N == 0) {
        return 0;
    }
    if (N == 1) {
        return 1;
    }
    int[] f = new int[N + 1];
    f[0] = 0;
    f[1] = 1;
    for (int i = 2; i <= N; i++) {
        f[i] = f[i - 1] + f[i - 2];
    }
    return f[N];
}
```  

### 88. Merge Sorted Array
1. 使用三个指针，m - 1, n - 1, m + n - 1，从右往左更新
```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int i = m - 1;
    int j = n - 1;
    int k = m + n - 1;
    while (i >= 0 && j >= 0) {
        if (nums1[i] < nums2[j]) {
            nums1[k] = nums2[j];
            j--;
            k--;
        } else {
            nums1[k] = nums1[i];
            i--;
            k--;
        }
    }
    while (j >= 0) {
        nums1[k] = nums2[j];
        k--;
        j--;
    }
}
```  

### 283. Move Zeroes
1. 双指针，更新右指针，只有在右值不为0的时候交换左右值，同时更新左指针
```java
public void moveZeroes(int[] nums) {
    int l = 0;
    int r = 0;
    while (r < nums.length) {
        if (nums[r] != 0) {
            int tmp = nums[r];
            nums[r] = nums[l];
            nums[l] = tmp;
            l++;
        }
        r++;
    }
}
```  

### 167. Two Sum || - Input array is sorted
1. 双指针，一左一右
```java
public int[] twoSum(int[] numbers, int target) {
    if (numbers.length < 2) {
        return new int[2];
    }
    int left = 0;
    int right = numbers.length - 1;
    while (left < right) {
        int sum = numbers[left] + numbers[right];
        if (sum == target) {
            return new int[] {left + 1, right + 1};
        } else if (sum > target) {
            right--;
        } else {
            left++;
        }
    }
    return new int[2];
}
```  

### 268. Missing Number
1. 求和相减, 为了防止overflow, 减的操作提前执行。
```java
public int missingNumber(int[] nums) {
    int res = 0;
    for (int i = 0; i < nums.length; i++) {
        res += i - nums[i] + 1; // the number after the missing one will have zero
    }
    return res;
}
```  

### 26. Remove Duplicates from Sorted Array
1. Two Pointer, 如果右值不等于左值，左值右移一位，更新左值
```java
public int removeDuplicates(int[] nums) {
    int l = 0;
    for (int r = 1; r < nums.length; r++) {
        if (nums[r] != nums[l]) {
            l++;
            nums[l] = nums[r];
        }
    }
    return l + 1;
}
```  

### 977. Squares of Sorted Array
1. O(nLogn), 遍历数组，求平方，再sort
2. O(n), Two Pointer, 一左一右
```java
public int[] sortedSquares(int[] A) {
    int[] res = new int[A.length];
    if (A.length == 0) {
        return res;
    }

    int l = 0;
    int r = A.length - 1;
    int p = res.length - 1;
    while (l <= r) {
        int i = A[l] * A[l];
        int j = A[r] * A[r];
        if (i > j) {
            res[p] = i;
            p--;
            l++;
        } else {
            res[p] = j;
            p--;
            r--;
        }
    }
    return res;
}
```

## Medium
### 15. Three Sum
1. Brute Force, three level for loop O(n^3)
2. Sort数组，第一个数One Pass，二三用双指针
```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length < 3) {
        return res;
    }
    
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) {
            continue;
        }
        
        int l = i + 1;
        int r = nums.length - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                l++;
                r--;
                while (l < r && nums[l] == nums[l - 1]) {
                    l++;
                }
            } else if (sum > 0) {
                r--;
            } else {
                l++;
            }
        }
    }
    return res;
}
```  

### 56. Merge Intervals
1. Sort Interval的最小值, 使用interval[0]记录区域，判断下一个区域的起点和当前区域的终点，  
重合则更新当前区域终点，否则更新当前区域两端，使用comparator提高效率
```java
public int[][] merge(int[][] intervals) {
    if (intervals == null || intervals.length == 0) {
        return intervals;
    }
    List<int[]> res = new ArrayList<int[]>();
    Arrays.sort(intervals, new IntervalsComparator());
    int[] curInterval = intervals[0];
    res.add(curInterval);
    for (int[] interval : intervals) {
        if (curInterval[1] >= interval[0]) {
            curInterval[1] = Math.max(interval[1], curInterval[1]);
        } else {
            curInterval = interval;
            res.add(curInterval);
        }
    }
    return res.toArray(new int[res.size()][2]);
}

private class IntervalsComparator implements Comparator<int[]> {
    public int compare(int[] a, int[] b) {
        return a[0] - b[0];
    }
}
```

### 33. Search in Rotated Sorted Array
1. Binary Search, 判断中心点在哪个区间, 判断是否在线性的区间内
```java
public int search(int[] nums, int target) {
    int len = nums.length, left = 0, right = len - 1;
    if (left > right)
        return -1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target)
            return mid;
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid])
                right = mid - 1;
            else
                left = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[right])
                left = mid + 1;
            else
                right = mid - 1;
        }
    }
    return -1;
}
```  

### 283. Product of Array Except Self
1. O(n) time, O(1) space, left -> right计算每个点左边的数的乘积,  
right -> left计算每个点右边数的乘积。
```java
public int[] productExceptSelf(int[] nums) {
    int[] res = new int[nums.length];
    res[0] = 1;
    for (int i = 1; i < nums.length; i++) {
        res[i] = nums[i - 1] * res[i - 1];
    }
    int r = 1;
    for (int i = nums.length - 1; i >= 0; i--) {
        res[i] = res[i] * r;
        r *= nums[i];
    }
    return res;
}
```  

### 11. Container With Most Water
1. 双指针, 一左一右, 两个指针的值比大小，更新指针
```java
public int maxArea(int[] height) {
    int l = 0;
    int r = height.length - 1;
    int max = 0;
    while (l < r) {
        max = Math.max(max, (r - l)*(Math.min(height[l], height[r])));
        if (height[l] > height[r]) {
            r--;
        } else {
            l++;
        }
    }
    return max;
}
```  

