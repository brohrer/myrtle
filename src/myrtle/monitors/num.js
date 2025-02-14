export function arrayMin(arr)
{
  let min_val = 1e20;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min_val){
      min_val = arr[i];
    }
  }
  return min_val
}

export function arrayMax(arr)
{
  let max_val = -1e20;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max_val){
      max_val = arr[i];
    }
  }
  return max_val
}
