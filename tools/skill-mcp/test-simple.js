#!/usr/bin/env node

/**
 * Simple test script for Skill MCP Server
 * Uses stdin/stdout directly instead of MCP client
 */

import { spawn } from "child_process";
import { createInterface } from "readline";

async function testSkillMcp() {
  console.log("Testing Skill MCP Server...\n");
  
  // Start the MCP server
  const serverProcess = spawn("node", ["tools/skill-mcp/index.js"], {
    stdio: ["pipe", "pipe", "pipe"],
    cwd: process.cwd()
  });
  
  // Create readline interface for stdout
  const rl = createInterface({
    input: serverProcess.stdout,
    crlfDelay: Infinity
  });
  
  // Collect responses
  const responses = [];
  
  rl.on("line", (line) => {
    try {
      const response = JSON.parse(line);
      responses.push(response);
    } catch (e) {
      // Ignore non-JSON output
    }
  });
  
  // Wait for server to start
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Test 1: List tools
  console.log("📋 Test 1: Listing tools...");
  serverProcess.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "tools/list"
  }) + "\n");
  
  // Wait for response
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  if (responses.length > 0) {
    const toolsResponse = responses[responses.length - 1];
    if (toolsResponse.result && toolsResponse.result.tools) {
      console.log(`✅ Found ${toolsResponse.result.tools.length} tools`);
      toolsResponse.result.tools.forEach(tool => {
        console.log(`   - ${tool.name}`);
      });
    }
  }
  
  // Test 2: List skills
  console.log("\n📚 Test 2: Listing skills...");
  serverProcess.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 2,
    method: "tools/call",
    params: {
      name: "list_skills",
      arguments: {}
    }
  }) + "\n");
  
  // Wait for response
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  if (responses.length > 1) {
    const skillsResponse = responses[responses.length - 1];
    if (skillsResponse.result && skillsResponse.result.content) {
      const skills = JSON.parse(skillsResponse.result.content[0].text);
      console.log(`✅ Found ${skills.length} skills`);
      console.log("   First 5 skills:");
      skills.slice(0, 5).forEach(skill => {
        console.log(`   - ${skill.name}`);
      });
    }
  }
  
  // Test 3: Get specific skill
  console.log("\n🔍 Test 3: Getting 'api-and-interface-design' skill...");
  serverProcess.stdin.write(JSON.stringify({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: {
      name: "get_skill",
      arguments: { name: "api-and-interface-design" }
    }
  }) + "\n");
  
  // Wait for response
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  if (responses.length > 2) {
    const getResponse = responses[responses.length - 1];
    if (getResponse.result && getResponse.result.content) {
      const skill = JSON.parse(getResponse.result.content[0].text);
      console.log(`✅ Skill found: ${skill.name}`);
      console.log(`   Description: ${skill.description.substring(0, 100)}...`);
      console.log(`   Source: ${skill.source}`);
    }
  }
  
  // Clean up
  serverProcess.kill();
  rl.close();
  
  console.log("\n✅ All tests completed!");
}

testSkillMcp().catch(console.error);
