import { app } from "/scripts/app.js";

const SPECIAL_SEED_RANDOM = -1;
const SPECIAL_SEED_INCREMENT = -2;
const SPECIAL_SEED_DECREMENT = -3;
const MAX_SEED = 1125899906842624;

function generateRandomSeed() {
  return Math.floor(Math.random() * MAX_SEED) + 1;
}

const seedNodes = new Set();
let queuePromptHooked = false;

function processSeedNodes() {
  for (const node of seedNodes) {
    if (node.mode === LiteGraph.NEVER) continue;

    const seedWidget = node.widgets?.find((w) => w.name === "seed");
    if (!seedWidget) continue;

    const inputSeed = Number(seedWidget.value);
    let seedToUse = inputSeed;

    if (inputSeed === SPECIAL_SEED_INCREMENT || inputSeed === SPECIAL_SEED_DECREMENT) {
      if (typeof node.lastSeed === "number") {
        if (inputSeed === SPECIAL_SEED_INCREMENT) {
          seedToUse = node.lastSeed + 1;
        } else {
          seedToUse = node.lastSeed - 1;
        }
      } else {
        seedToUse = generateRandomSeed();
      }

      seedWidget.value = seedToUse;
      node.lastSeed = seedToUse;

      if (node.lastSeedValueWidget) {
        node.lastSeedValueWidget.value = `Last Seed: ${seedToUse}`;
      }
      if (node.lastSeedButton) {
        node.lastSeedButton.label = `\u267B\uFE0F ${seedToUse}`;
      }
    }
  }
}

function hookQueuePrompt() {
  if (queuePromptHooked) return;
  queuePromptHooked = true;

  const origQueuePrompt = app.queuePrompt?.bind?.(app);
  if (origQueuePrompt) {
    app.queuePrompt = function (...args) {
      processSeedNodes();
      return origQueuePrompt(...args);
    };
  }
}

app.registerExtension({
  name: "sfnodes.Seed",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SFSeed") return;

    hookQueuePrompt();

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      if (message && message["SEED"] !== undefined) {
        const seed = Array.isArray(message["SEED"]) ? message["SEED"][0] : message["SEED"];
        this.lastSeed = Number(seed);
        if (this.lastSeedValueWidget) {
          this.lastSeedValueWidget.value = `Last Seed: ${this.lastSeed}`;
        }
        if (this.lastSeedButton) {
          this.lastSeedButton.label = `\u267B\uFE0F ${this.lastSeed}`;
        }
      }
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      seedNodes.add(this);

      const seedWidget = this.widgets?.find((w) => w.name === "seed");
      if (seedWidget) {
        seedWidget.value = SPECIAL_SEED_RANDOM;

        const cagIndex = this.widgets.findIndex((w) => w.name === "control_after_generate");
        if (cagIndex >= 0) {
          this.widgets.splice(cagIndex, 1);
        }
      }

      this.addWidget("button", "\uD83C\uDFB2 Randomize Each Time", "", () => {
        if (seedWidget) {
          seedWidget.value = SPECIAL_SEED_RANDOM;
        }
      }, { serialize: false });

      this.addWidget("button", "\uD83C\uDFB2 New Fixed Random", "", () => {
        if (seedWidget) {
          seedWidget.value = generateRandomSeed();
        }
      }, { serialize: false });

      this.lastSeedButton = this.addWidget("button", "\u267B\uFE0F Use Last Queued Seed", "", () => {
        if (seedWidget && typeof this.lastSeed === "number") {
          seedWidget.value = this.lastSeed;
          this.lastSeedButton.label = "\u267B\uFE0F (Use Last Queued Seed)";
        }
      }, { serialize: false });

      this.lastSeedValueWidget = null;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      seedNodes.delete(this);
      onRemoved?.apply(this, arguments);
    };
  },
});
