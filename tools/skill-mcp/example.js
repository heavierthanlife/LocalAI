#!/usr/bin/env node

/**
 * Example: How to use the Skill MCP Server
 * This script demonstrates how to invoke skills programmatically
 */

import { spawn } from "child_process";

async function invokeSkillExample() {
  console.log("🎯 Skill MCP Server Usage Example\n");
  
  // Start the MCP server
  const server = spawn("node", ["tools/skill-mcp/index.js"], {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: process.cwd()
  });
  
  // Helper function to send request and get response
  async function sendRequest(request) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Request timeout"));
      }, 10000);
      
      server.stdout.once("data", (data) => {
        clearTimeout(timeout);
        try {
          resolve(JSON.parse(data.toString()));
        } catch (e) {
          reject(new Error("Invalid JSON response"));
        }
      });
      
      server.stdin.write(JSON.stringify(request) + "\n");
    });
  }
  
  // Wait for server to start
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  try {
    // Example 1: List all skills
    console.log("📚 Example 1: List all available skills");
    const listResponse = await sendRequest({
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: "list_skills",
        arguments: {}
      }
    });
    
    const skills = JSON.parse(listResponse.result.content[0].text);
    console.log(`Found ${skills.length} skills`);
    console.log("Sample skills:");
    skills.slice(0, 3).forEach(skill => {
      console.log(`  - ${skill.name}: ${skill.description.substring(0, 60)}...`);
    });
    console.log("");
    
    // Example 2: Get a specific skill
    console.log("🔍 Example 2: Get 'api-and-interface-design' skill details");
    const getResponse = await sendRequest({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/call",
      params: {
        name: "get_skill",
        arguments: { name: "api-and-interface-design" }
      }
    });
    
    const skill = JSON.parse(getResponse.result.content[0].text);
    console.log(`Skill: ${skill.name}`);
    console.log(`Description: ${skill.description}`);
    console.log(`Source: ${skill.source}`);
    console.log(`Content preview: ${skill.content.substring(0, 200)}...`);
    console.log("");
    
    // Example 3: Invoke a skill
    console.log("🚀 Example 3: Invoke 'api-and-interface-design' skill with arguments");
    const invokeResponse = await sendRequest({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: {
        name: "invoke_skill",
        arguments: {
          name: "api-and-interface-design",
          args: {
            task: "Design a REST API for a todo list application",
            context: "Using Flask and PostgreSQL"
          }
        }
      }
    });
    
    const invocation = JSON.parse(invokeResponse.result.content[0].text);
    console.log(`Invoked skill: ${invocation.skill}`);
    console.log(`Instructions: ${invocation.instructions}`);
    console.log(`Arguments: ${JSON.stringify(invocation.args, null, 2)}`);
    console.log("");
    
    console.log("✅ All examples completed successfully!");
    
  } catch (error) {
    console.error("❌ Error:", error.message);
  } finally {
    // Clean up
    server.kill();
  }
}

invokeSkillExample().catch(console.error);
