import { ChatCompletionRole, ChatCompletionUserMessageParam } from "openai/resources";
import { AzureOpenAIChatCompletionsFormatter } from "../../src/azure-openai-chat-completions-formatter.js";
import { z } from "zod";
import { Kanuni } from "kanuni";
import { jest } from '@jest/globals';

describe("AzureOpenAIChatCompletionsFormatter", () => {
  const schemaName = "TestSchema";

  describe("Basic functionality", () => {
    it("should format a basic query with default instructions formatter (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }), schemaName)
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[0].content).toContain("This is a test prompt");
      expect(result.response_format).toBeDefined();
      if (result.response_format) {
        expect(result.response_format.type).toBe("json_schema");
        if (result.response_format.type === "json_schema") {
          expect(result.response_format.json_schema.name).toBe(schemaName);
        }
      }
    });

    it("should format a basic query with default instructions formatter (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[0].content).toContain("This is a test prompt");
      expect(result.response_format).toBeUndefined();
    });

    it("should use the provided instructionsRole and instructionsFormatter (JSON output)", () => {
      const customFormatter = jest.fn(() => "custom instructions");
      const formatter = new AzureOpenAIChatCompletionsFormatter({
        instructionsRole: "developer",
        instructionsFormatter: customFormatter,
      });
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[0].role).toBe("developer");
      expect(result.messages[0].content).toBe("custom instructions");
      expect(customFormatter).toHaveBeenCalledWith(query);
    });

    it("should use the provided instructionsRole and instructionsFormatter (text output)", () => {
      const customFormatter = jest.fn(() => "custom instructions");
      const formatter = new AzureOpenAIChatCompletionsFormatter({
        instructionsRole: "developer",
        instructionsFormatter: customFormatter,
      });
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[0].role).toBe("developer");
      expect(result.messages[0].content).toBe("custom instructions");
      expect(customFormatter).toHaveBeenCalledWith(query);
      expect(result.response_format).toBeUndefined();
    });

    it("should handle queries with no memory contents (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(1);
      expect(result.messages[0].role).toBe("system");
    });

    it("should handle queries with no memory contents (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(1);
      expect(result.messages[0].role).toBe("system");
      expect(result.response_format).toBeUndefined();
    });
  });

  describe("Role mapping", () => {
    it("should correctly map roles using a custom roleMapper (JSON output)", () => {
      const roleMapper = jest.fn((role: ChatCompletionRole) => (role === "user" ? "assistant" : 'user'));
      const formatter = new AzureOpenAIChatCompletionsFormatter({ roleMapper });
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", () => "hi" )
        )
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(roleMapper).toHaveBeenCalledWith("user", undefined);
      expect(result.messages[1].role).toBe("assistant");
      expect(result.messages[1].content).toBe("hi");
    });

    it("should correctly map roles using a custom roleMapper (text output)", () => {
      const roleMapper = jest.fn((role: ChatCompletionRole) => (role === "user" ? "assistant" : 'user'));
      const formatter = new AzureOpenAIChatCompletionsFormatter({ roleMapper });
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", () => "hi" )
        )
        .build({});
      const result = formatter.format(query);
      
      expect(roleMapper).toHaveBeenCalledWith("user", undefined);
      expect(result.messages[1].role).toBe("assistant");
      expect(result.messages[1].content).toBe("hi");
      expect(result.response_format).toBeUndefined();
    });

    it("should use identity role mapper by default", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", () => "hi")
          .utterance("assistant", () => "hello")
        )
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(3); // system + user + assistant
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[2].role).toBe("assistant");
    });

    it("should throw an error if a memory item maps to an invalid utterance role", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({
        // @ts-expect-error - testing invalid role mapping
        roleMapper: () => "system",
      });
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", "bob", () => "hi"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      expect(() => formatter.format(query)).toThrow(/Only user and assistant utterances are supported/);
    });

    it("should throw an error for unknown roles in identity mapper", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("unknown_role" as any, () => "hi"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      expect(() => formatter.format(query)).toThrow(/Unknown role: unknown_role/);
    });
  });

  describe("Memory handling", () => {
    it("should handle single user utterance (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "hi"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(2);
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[1].content).toBe("hi");
    });

    it("should handle single user utterance (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "hi"))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(2);
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[1].content).toBe("hi");
      expect(result.response_format).toBeUndefined();
    });

    it("should handle single assistant utterance", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("assistant", () => "hello"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(2);
      expect(result.messages[1].role).toBe("assistant");
      expect(result.messages[1].content).toBe("hello");
    });

    it("should handle multiple consecutive utterances from different roles", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", () => "hi")
          .utterance("assistant", () => "hello")
          .utterance("user", () => "how are you?")
        )
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(4); // system + user + assistant + user
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[1].content).toBe("hi");
      expect(result.messages[2].role).toBe("assistant");
      expect(result.messages[2].content).toBe("hello");
      expect(result.messages[3].role).toBe("user");
      expect(result.messages[3].content).toBe("how are you?");
    });

    it("should handle queries with named utterances", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", "bob", () => "hi"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[1].role).toBe("user");
      expect((result.messages[1] as ChatCompletionUserMessageParam).name).toBe("bob");
      expect(result.messages[1].content).toBe("hi");
    });

    it("should handle utterances without names", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "hi"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages[1].role).toBe("user");
      expect((result.messages[1] as ChatCompletionUserMessageParam).name).toBeUndefined();
      expect(result.messages[1].content).toBe("hi");
    });
  });

  describe("Tool handling", () => {
    it("should return empty tools array when no tools are provided", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.tools).toEqual([]);
    });

    it("should format tools correctly when provided", () => {
      // TODO: This test needs to be implemented once we understand how Kanuni handles tools
      // For now, we'll test that the formatter handles undefined tools correctly
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      const result = formatter.format(query);
      
      // When no tools are provided, should return empty array
      expect(result.tools).toEqual([]);
    });

    // TODO: Add tests for tool calls and tool call results when Kanuni supports them in memory
    // This would require understanding how Kanuni structures tool calls in memory
  });

  describe("Output type handling", () => {
    it("should handle text output correctly", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "hello"))
        .build({});
      const result = formatter.format(query);
      
      expect(result.response_format).toBeUndefined();
      expect(result.messages.length).toBe(2);
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[1].role).toBe("user");
    });

    it("should handle JSON output correctly", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "hello"))
        .outputJson(z.strictObject({
          response: z.string().describe("The response field"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.response_format).toBeDefined();
      expect(result.response_format?.type).toBe("json_schema");
      expect(result.messages.length).toBe(2);
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[1].role).toBe("user");
    });

    it("should throw error for unknown output types", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = {
        prompt: { type: "prompt", contents: [] },
        output: { type: "unknown-output-type" },
      } as any;
      
      expect(() => formatter.format(query)).toThrow(/Unknown output type: unknown-output-type/);
    });
  });

  describe("JSON Schema handling", () => {
    it("should include the correct JSON schema in response_format", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }), schemaName)
        .build({});
      const result = formatter.format(query);
      
      expect(result.response_format).toBeDefined();
      if (result.response_format) {
        expect(result.response_format.type).toBe("json_schema");
        if (result.response_format.type === "json_schema") {
          expect(result.response_format.json_schema.name).toBe(schemaName);
          expect(result.response_format.json_schema.strict).toBe(true);
          expect(result.response_format.json_schema.schema).toBeDefined();
        }
      }
    });

    it("should handle JSON schema without explicit name", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.response_format).toBeDefined();
      if (result.response_format) {
        expect(result.response_format.type).toBe("json_schema");
      }
    });
  });

  describe("Complex memory scenarios", () => {
    it("should handle empty memory gracefully (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m) // empty memory
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(1);
      expect(result.messages[0].role).toBe("system");
    });

    it("should handle empty memory gracefully (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m) // empty memory
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(1);
      expect(result.messages[0].role).toBe("system");
      expect(result.response_format).toBeUndefined();
    });

    it("should handle consecutive utterances from the same role", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", () => "First message")
          .utterance("user", () => "Second message")
          .utterance("assistant", () => "Response to first")
          .utterance("assistant", () => "Response to second")
        )
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      // Should create separate messages for each utterance
      expect(result.messages.length).toBe(5); // system + 4 utterances
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[1].content).toBe("First message");
      expect(result.messages[2].role).toBe("user");
      expect(result.messages[2].content).toBe("Second message");
      expect(result.messages[3].role).toBe("assistant");
      expect(result.messages[3].content).toBe("Response to first");
      expect(result.messages[4].role).toBe("assistant");
      expect(result.messages[4].content).toBe("Response to second");
    });

    it("should handle mixed named and unnamed utterances", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m
          .utterance("user", "alice", () => "Hi from Alice")
          .utterance("user", () => "Anonymous user message")
          .utterance("assistant", () => "Assistant response")
        )
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      const result = formatter.format(query);
      
      expect(result.messages.length).toBe(4); // system + 3 utterances
      expect((result.messages[1] as ChatCompletionUserMessageParam).name).toBe("alice");
      expect((result.messages[2] as ChatCompletionUserMessageParam).name).toBeUndefined();
      expect(result.messages[3].role).toBe("assistant");
    });
  });

  describe("Edge cases", () => {
    it("should handle formatter configuration with all options (JSON output)", () => {
      const customInstructionsFormatter = jest.fn(() => "custom formatted instructions");
      const customRoleMapper = jest.fn((role: string) => role === "user" ? "user" : "assistant");
      
      const formatter = new AzureOpenAIChatCompletionsFormatter({
        instructionsRole: "developer",
        instructionsFormatter: customInstructionsFormatter,
        roleMapper: customRoleMapper,
      });
      
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "test"))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      const result = formatter.format(query);
      
      expect(customInstructionsFormatter).toHaveBeenCalledWith(query);
      expect(customRoleMapper).toHaveBeenCalledWith("user", undefined);
      expect(result.messages[0].role).toBe("developer");
      expect(result.messages[0].content).toBe("custom formatted instructions");
    });

    it("should handle formatter configuration with all options (text output)", () => {
      const customInstructionsFormatter = jest.fn(() => "custom formatted instructions");
      const customRoleMapper = jest.fn((role: string) => role === "user" ? "user" : "assistant");
      
      const formatter = new AzureOpenAIChatCompletionsFormatter({
        instructionsRole: "developer",
        instructionsFormatter: customInstructionsFormatter,
        roleMapper: customRoleMapper,
      });
      
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => "test"))
        .build({});
      
      const result = formatter.format(query);
      
      expect(customInstructionsFormatter).toHaveBeenCalledWith(query);
      expect(customRoleMapper).toHaveBeenCalledWith("user", undefined);
      expect(result.messages[0].role).toBe("developer");
      expect(result.messages[0].content).toBe("custom formatted instructions");
      expect(result.response_format).toBeUndefined();
    });

    it("should handle empty format params (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      // Test that undefined params work the same as empty object
      const result1 = formatter.format(query);
      const result2 = formatter.format(query, {});
      const result3 = formatter.format(query, undefined as any);
      
      expect(result1).toEqual(result2);
      expect(result1).toEqual(result3);
    });

    it("should handle empty format params (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .build({});
      
      // Test that undefined params work the same as empty object
      const result1 = formatter.format(query);
      const result2 = formatter.format(query, {});
      const result3 = formatter.format(query, undefined as any);
      
      expect(result1).toEqual(result2);
      expect(result1).toEqual(result3);
      expect(result1.response_format).toBeUndefined();
    });

    it("should handle very long conversation histories", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const memoryBuilder = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => {
          let builder = m;
          // Create a long conversation
          for (let i = 0; i < 50; i++) {
            builder = builder
              .utterance("user", () => `User message ${i}`)
              .utterance("assistant", () => `Assistant response ${i}`);
          }
          return builder;
        });
      
      const query = memoryBuilder
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      const result = formatter.format(query);
      
      // Should have system message + 100 conversation messages (50 user + 50 assistant)
      expect(result.messages.length).toBe(101);
      expect(result.messages[0].role).toBe("system");
      expect(result.messages[1].role).toBe("user");
      expect(result.messages[1].content).toBe("User message 0");
      expect(result.messages[2].role).toBe("assistant");
      expect(result.messages[2].content).toBe("Assistant response 0");
      expect(result.messages[100].content).toBe("Assistant response 49");
    });

    it("should preserve message content exactly as provided (JSON output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const specialContent = "Special chars: 🚀 \n\t\"quotes\" and 'apostrophes' and @mentions #hashtags $variables";
      
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => specialContent))
        .outputJson(z.strictObject({
          aField: z.string().describe("A field in the output schema"),
        }))
        .build({});
      
      const result = formatter.format(query);
      
      expect(result.messages[1].content).toBe(specialContent);
    });

    it("should preserve message content exactly as provided (text output)", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const specialContent = "Special chars: 🚀 \n\t\"quotes\" and 'apostrophes' and @mentions #hashtags $variables";
      
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .memory(m => m.utterance("user", () => specialContent))
        .build({});
      
      const result = formatter.format(query);
      
      expect(result.messages[1].content).toBe(specialContent);
      expect(result.response_format).toBeUndefined();
    });

    it("should handle complex JSON schema with nested objects", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      const complexSchema = z.strictObject({
        user: z.strictObject({
          name: z.string(),
          age: z.number(),
          preferences: z.array(z.string())
        }),
        metadata: z.strictObject({
          timestamp: z.string(),
          version: z.number()
        })
      });
      
      const query = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(complexSchema, "ComplexResponse")
        .build({});
      
      const result = formatter.format(query);
      
      expect(result.response_format).toBeDefined();
      if (result.response_format && result.response_format.type === "json_schema") {
        expect(result.response_format.json_schema.name).toBe("ComplexResponse");
        expect(result.response_format.json_schema.strict).toBe(true);
        expect(result.response_format.json_schema.schema).toBeDefined();
      }
    });

    it("should handle switching between output types with same formatter instance", () => {
      const formatter = new AzureOpenAIChatCompletionsFormatter({});
      
      // Test text output query
      const textQuery = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .build({});
      const textResult = formatter.format(textQuery);
      
      // Test JSON output query
      const jsonQuery = Kanuni.newQuery()
        .prompt(p => p.paragraph`This is a test prompt`)
        .outputJson(z.strictObject({
          response: z.string(),
        }))
        .build({});
      const jsonResult = formatter.format(jsonQuery);
      
      // Verify text output has no response_format
      expect(textResult.response_format).toBeUndefined();
      expect(textResult.messages[0].content).toContain("This is a test prompt");
      
      // Verify JSON output has response_format
      expect(jsonResult.response_format).toBeDefined();
      expect(jsonResult.response_format?.type).toBe("json_schema");
      expect(jsonResult.messages[0].content).toContain("This is a test prompt");
    });
  });
});
