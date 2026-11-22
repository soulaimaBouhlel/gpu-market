require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.24",
    settings: { optimizer: { enabled: true, runs: 200 }, viaIR: true },
  },
  networks: {
    // Ganache running on this machine (VM 1). Accounts come from Ganache itself.
    ganache: {
      url: process.env.GANACHE_URL || "http://127.0.0.1:8545",
      chainId: 1337,
    },
  },
};
