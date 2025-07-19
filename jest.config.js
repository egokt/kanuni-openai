/** @type {import('ts-jest').JestConfigWithTsJest} **/
export default {
  testEnvironment: "node",
  transform: {
    "\\.[jt]sx?$": [
      "ts-jest",
      {
        useESM: true,
        tsconfig: "./tests/tsconfig.json",
      },
    ],
  },
  moduleNameMapper: {
    "(.+)\\.js": "$1",
  },
  extensionsToTreatAsEsm: [".ts"],
};
