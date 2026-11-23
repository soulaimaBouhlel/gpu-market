// Deploy ModelRegistry + JobMarket to the running Hardhat node and write
// deployments/localhost.json (addresses + ABIs) for the Python agents.
//
// Run on VM 1 (the chain VM), with the blockchain node already running:
//   Ganache:       npx hardhat run scripts/deploy.js --network ganache
//   Hardhat node:  npx hardhat run scripts/deploy.js --network localhost
const { ethers, network, artifacts } = require("hardhat");
const fs = require("fs");

async function main() {
  // The chain must produce blocks even when idle, so that "wait N blocks
  // before the challenge" works. Ganache: started with --miner.blockTime 2.
  // Hardhat node: switch interval mining on here (Hardhat-only RPC method).
  try {
    await network.provider.send("evm_setIntervalMining", [2000]);
    console.log("interval mining enabled (Hardhat node)");
  } catch (e) {
    console.log("not a Hardhat node: relying on the node's own block time (Ganache --miner.blockTime)");
  }

  const revealWindow = Number(process.env.REVEAL_WINDOW || 120);   // seconds
  const Registry = await ethers.getContractFactory("ModelRegistry");
  const registry = await Registry.deploy();
  await registry.waitForDeployment();

  const Market = await ethers.getContractFactory("JobMarket");
  const market = await Market.deploy(await registry.getAddress(), revealWindow);
  await market.waitForDeployment();

  const out = {
    network: network.name,
    revealWindow,
    ModelRegistry: {
      address: await registry.getAddress(),
      abi: (await artifacts.readArtifact("ModelRegistry")).abi,
    },
    JobMarket: {
      address: await market.getAddress(),
      abi: (await artifacts.readArtifact("JobMarket")).abi,
    },
  };
  fs.mkdirSync("deployments", { recursive: true });
  fs.writeFileSync("deployments/localhost.json", JSON.stringify(out, null, 2));
  console.log("ModelRegistry:", out.ModelRegistry.address);
  console.log("JobMarket:    ", out.JobMarket.address);
  console.log("reveal window:", revealWindow, "s");
  console.log("written deployments/localhost.json");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
