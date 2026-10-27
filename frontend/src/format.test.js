import { describe, expect, it } from "vitest";
import { clamp, clock, level, num, pct } from "./format";

describe("format helpers", () => {
  it("formats clock times", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(125)).toBe("2:05");
    expect(clock(null)).toBe("–:––");
  });
  it("never shows NaN", () => {
    expect(pct(NaN)).toBe("–");
    expect(num(undefined)).toBe("–");
    expect(pct(0.256, 1)).toBe("25.6%");
  });
  it("clamps levels and values", () => {
    expect(level(9).name).toBe("CRITICAL");
    expect(level(-3).name).toBe("CLEAR");
    expect(clamp(5, 0, 1)).toBe(1);
  });
});
