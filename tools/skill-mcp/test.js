#!/usr/bin/env node

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { spawn } from "child_process";

async function testSkillMcp() {
  console.log("Testing Skill MCP Server...\n");
  
  // Start the MCP server
  const serverProcess = spawn("node", ["tools/skill-mcp/index.js"], {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: process.cwd()
  });
  
  // Create client transport
  const transport = new StdioClientTransport({
    stdin: serverProcess.stdin,
    stdout: serverProcess.stdout
  });
  
  // Create client
  const client = new Client({
    name: "test-client",
    version: "1.0.0"
  });
  
  try {
    // Connect to server
    await client.connect(transport);
    console.log("✅ Connected to Skill MCP server\n");
    
    // List available tools
    const tools = await client.listTools();
    console.log("📋 Available tools:");
    tools.tools.forEach(tool => {
      console.log(`  - ${tool.name}: ${tool.description}`);
    });
    console.log("");
    
    // List available skills
    console.log("📚 Listing skills...");
    const listResult = await client.callTool({
      name: "list_skills",
      arguments: {}
    });
    
    const skills = JSON.parse(listResult.content[0].text);
    console.log(`Found ${skills.length} skills\n`);
    
    // Show first 5 skills
    console.log("First 5 skills:");
    skills.slice(0, 5).forEach(skill => {
      console.log(`  - ${skill.name}: ${skill.description.substring(0, 80)}...`);
    });
    console.log("");
    
    // Get specific skill
    console.log("🔍 Getting 'api-and-interface-design' skill...");
    const getResult = await client.callTool({
      name: "get_skill",
      arguments: { name: "api-and-interface-design" }
    });
    
    const skill = JSON.parse(getResult.content[0].text);
    console.log(`Skill: ${skill.name}`);
    console.log(`Description: ${skill.description}`);
    console.log(`Source: ${skill.source}`);
    console.log(`Content length: ${skill.content.length} characters\n`);
    
    // Invoke skill
    console.log("🚀 Invoking 'api-and-interface-design' skill...");
    const invokeResult = await client.callTool({
      name: "invoke_skill",
      arguments: {
        name: "api-and-interface-design",
        args: {
          task: "Design a REST API for a todo list application"
        }
      }
    });
    
    const invocation = JSON.parse(invokeResult.content[0].text);
    console.log(`Skill invoked: ${invocation.skill}`);
    console.log(`Instructions: ${invocation.instructions}`);
    console.log(`Args: ${JSON.stringify(invocation.args)}`);
    console.log("");
    
    console.log("✅ All tests passed!");
    
  } catch (error) {
    console.error("❌ Test failed:", error.message);
  } finally {
    // Clean up
    serverProcess.kill();
    await client.close();
  }
}

testSkillMcp().catch(console.error);
