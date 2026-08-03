#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { glob } from "glob";
import fs from "fs-extra";
import path from "path";
import { fileURLToPath } from "url";
import { homedir } from "os";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Skill library paths
const SHARED_SKILLS_DIR = "D:\\AI_Tools\\shared-agent-infra\\skills";
const LOCAL_SKILLS_DIR = ".opencode/skills";

// Cache for loaded skills
let skillsCache = null;
let lastCacheUpdate = 0;
const CACHE_TTL = 60000; // 1 minute

/**
 * Load and parse all skills from configured directories
 */
async function loadSkills() {
  const now = Date.now();
  if (skillsCache && now - lastCacheUpdate < CACHE_TTL) {
    return skillsCache;
  }

  const skills = new Map();
  
  // Load from shared skills directory
  if (await fs.pathExists(SHARED_SKILLS_DIR)) {
    const sharedSkills = await loadSkillsFromDir(SHARED_SKILLS_DIR, "shared");
    for (const [name, skill] of sharedSkills) {
      if (!skills.has(name)) {
        skills.set(name, skill);
      }
    }
  }

  // Load from local skills directory
  const localSkillsPath = path.resolve(process.cwd(), LOCAL_SKILLS_DIR);
  if (await fs.pathExists(localSkillsPath)) {
    const localSkills = await loadSkillsFromDir(localSkillsPath, "local");
    for (const [name, skill] of localSkills) {
      skills.set(name, skill); // Local skills override shared
    }
  }

  skillsCache = skills;
  lastCacheUpdate = now;
  return skills;
}

/**
 * Load skills from a specific directory
 */
async function loadSkillsFromDir(dirPath, source) {
  const skills = new Map();
  
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith(".") || entry.name === "_old") {
        continue;
      }
      
      const skillDir = path.join(dirPath, entry.name);
      const skillMdPath = path.join(skillDir, "SKILL.md");
      
      if (await fs.pathExists(skillMdPath)) {
        try {
          const content = await fs.readFile(skillMdPath, "utf-8");
          const skill = parseSkill(content, entry.name, skillDir, source);
          if (skill) {
            skills.set(entry.name, skill);
          }
        } catch (error) {
          console.error(`Error loading skill ${entry.name}:`, error.message);
        }
      }
    }
  } catch (error) {
    console.error(`Error reading directory ${dirPath}:`, error.message);
  }
  
  return skills;
}

/**
 * Parse a SKILL.md file
 */
function parseSkill(content, name, dirPath, source) {
  const lines = content.split("\n");
  let description = "";
  let inFrontmatter = false;
  let frontmatterEnd = 0;
  
  // Parse frontmatter
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (i === 0 && line === "---") {
      inFrontmatter = true;
      continue;
    }
    
    if (inFrontmatter && line === "---") {
      inFrontmatter = false;
      frontmatterEnd = i;
      continue;
    }
    
    if (inFrontmatter && line.startsWith("description:")) {
      description = line.substring("description:".length).trim();
      // Remove quotes if present
      if ((description.startsWith('"') && description.endsWith('"')) ||
          (description.startsWith("'") && description.endsWith("'"))) {
        description = description.slice(1, -1);
      }
    }
  }
  
  // Extract the main content after frontmatter
  const mainContent = lines.slice(frontmatterEnd + 1).join("\n").trim();
  
  return {
    name,
    description: description || `Skill: ${name}`,
    content: mainContent,
    path: dirPath,
    source
  };
}

/**
 * List all available skills
 */
async function listSkills() {
  const skills = await loadSkills();
  return Array.from(skills.values()).map(skill => ({
    name: skill.name,
    description: skill.description,
    source: skill.source,
    path: skill.path
  }));
}

/**
 * Get a specific skill by name
 */
async function getSkill(name) {
  const skills = await loadSkills();
  return skills.get(name) || null;
}

/**
 * Create MCP server
 */
const server = new Server(
  {
    name: "skill-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_skills",
        description: "List all available skills from the skill library",
        inputSchema: {
          type: "object",
          properties: {},
          required: [],
        },
      },
      {
        name: "get_skill",
        description: "Get detailed information about a specific skill",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Name of the skill to retrieve",
            },
          },
          required: ["name"],
        },
      },
      {
        name: "invoke_skill",
        description: "Invoke a skill by name with optional arguments",
        inputSchema: {
          type: "object",
          properties: {
            name: {
              type: "string",
              description: "Name of the skill to invoke",
            },
            args: {
              type: "object",
              description: "Optional arguments to pass to the skill",
              additionalProperties: true,
            },
          },
          required: ["name"],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    switch (name) {
      case "list_skills": {
        const skills = await listSkills();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(skills, null, 2),
            },
          ],
        };
      }
      
      case "get_skill": {
        const skill = await getSkill(args.name);
        if (!skill) {
          return {
            content: [
              {
                type: "text",
                text: `Skill "${args.name}" not found`,
              },
            ],
            isError: true,
          };
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(skill, null, 2),
            },
          ],
        };
      }
      
      case "invoke_skill": {
        const skill = await getSkill(args.name);
        if (!skill) {
          return {
            content: [
              {
                type: "text",
                text: `Skill "${args.name}" not found`,
              },
            ],
            isError: true,
          };
        }
        
        // Return the skill content for the agent to use
        const response = {
          skill: skill.name,
          description: skill.description,
          content: skill.content,
          args: args.args || {},
          instructions: `You are now invoking the "${skill.name}" skill. Follow the instructions in the skill content below to complete the task. Use the provided arguments if any.`
        };
        
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(response, null, 2),
            },
          ],
        };
      }
      
      default:
        return {
          content: [
            {
              type: "text",
              text: `Unknown tool: ${name}`,
            },
          ],
          isError: true,
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// List available resources
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "skill://list",
        name: "List of all available skills",
        mimeType: "application/json",
      },
    ],
  };
});

// Read resource
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;
  
  if (uri === "skill://list") {
    const skills = await listSkills();
    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify(skills, null, 2),
        },
      ],
    };
  }
  
  throw new Error(`Unknown resource: ${uri}`);
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Skill MCP server running on stdio");
}

main().catch(console.error);
