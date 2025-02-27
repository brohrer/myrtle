const epsilon = 1e-10;
const huge = 1e20;

export function intervalSpacedArray(
  start,
  stop,
  interval,
  includeStop = False,
) {
  if (includeStop) {
    stop += epsilon;
  }
  let i = 0;
  let arr = [];
  for (let val = start; val < stop; val += interval) {
    arr[i] = val;
    i += 1;
  }
  return arr;
}

export function max(arr) {
  let max_val = -1 * huge;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max_val) {
      max_val = arr[i];
    }
  }
  return max_val;
}

export function min(arr) {
  let min_val = huge;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min_val) {
      min_val = arr[i];
    }
  }
  return min_val;
}

export function prettySpacedArray(extentA, extentB, minLength = 9) {
  let minValue = min([extentA, extentB]);
  let maxValue = max([extentA, extentB]);
  let maxInterval = (maxValue - minValue + epsilon) / (minLength - 1.5);
  let logInterval = Math.log10(maxInterval);
  let logIntervalBase = Math.floor(logInterval);
  let logIntervalRemainder = logInterval - logIntervalBase;

  // This gives a spacing like
  // 10, 20, 30, 40, ...
  // or
  // 0.007, 0.008. 0.009, 0.010, ...
  // -whole number last digit increments
  let interval = Math.pow(10.0, logIntervalBase);

  // This gives a spacing like
  // 20, 40, 60, 80, ...
  // or
  // 0.008, 0.010. 0.012, 0.014, ...
  // -even number last digit increments
  // log of .3 is a about factor of 2.
  if (logIntervalRemainder > 0.3) {
    interval *= 2.0;
  }

  // This gives a spacing like
  // 50, 100, 150, 200, ...
  // or
  // 0.075, 0.080. 0.085, 0.090, ...
  // -last digit increments of 5
  // log of .7 is about factor of 5.
  if (logIntervalRemainder > 0.7) {
    // Combined with the previous factor of 2, this makes a factor of 5.
    interval *= 2.5;
  }

  let lowestValue = interval * Math.round(minValue / interval);
  let highestValue = interval * Math.round((maxValue + epsilon) / interval);
  let valArray = intervalSpacedArray(lowestValue, highestValue, interval, true);
  let strArray = [];

  for (let i = 0; i < valArray.length; i++) {
    let precision = -1 * logIntervalBase;
    if (precision > 0) {
      strArray[i] = valArray[i].toFixed(precision);
    } else {
      let value = Math.round(valArray[i]);
      if (interval % 1000000 == 0 && value != 0 ) {
        strArray[i] = `${value / 1000000}M`;
      } else if (interval % 1000 == 0 && value != 0) {
        strArray[i] = `${value / 1000}K`;
      } else {
        strArray[i] = value;
      }
    }
  }
  return [valArray, strArray];
}

// Create an array covering the range from `start`
// up to (but not including) `stop`
export function range(start, stop) {
  return intervalSpacedArray(Math.round(start), Math.round(stop), 1.0, false);
}

// Create an array covering the range from `start`
// up to and including `stop`
export function rangeInclusive(start, stop) {
  return intervalSpacedArray(Math.round(start), Math.round(stop), 1.0, true);
}

export function zeros(len) {
  return new Array(len).fill(0);
}
