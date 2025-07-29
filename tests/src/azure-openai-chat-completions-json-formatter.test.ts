import { ChatCompletionRole, ChatCompletionUserMessageParam } from "openai/resources";
import { AzureOpenAIChatCompletionsJsonFormatter } from "../../src/azure-openai-chat-completions-json-formatter.js";
import { z } from "zod";
import { Kanuni } from "kanuni";
import { jest } from '@jest/globals';

describe("AzureOpenAIChatCompletionsJsonFormatter", () => {
  const schemaName = "TestSchema";

  it("should format a basic query with default instructions formatter", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
    // prettier-ignore
    const query = Kanuni.newQuery()
      .prompt(p => p.paragraph`This is a test prompt`)
      .outputJson(z.strictObject({
        aField: z.string().describe("A field in the output schema"),
      }), schemaName)
      .build({});
    const result = formatter.format(query);
    expect(result.messages[0].role).toBe("system");
    expect(result.response_format).toBeDefined();
    if (result.response_format) {
      expect(result.response_format.type).toBe("json_schema");
      if (result.response_format.type === "json_schema") {
        expect(result.response_format.json_schema.name).toBe(schemaName);
      }
    }
  });

  it("should use the provided instructionsRole and instructionsFormatter", () => {
    const customFormatter = jest.fn(() => "custom instructions");
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({
      instructionsRole: "developer",
      instructionsFormatter: customFormatter,
    });
    // prettier-ignore
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

  it("should correctly map roles using a custom roleMapper", () => {
    const roleMapper = jest.fn((role: ChatCompletionRole) => (role === "user" ? "assistant" : role));
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({ roleMapper });
    // prettier-ignore
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

  it("should throw an error if the query output type is not 'output-json'", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
    // const badQuery = { output: { type: "output-text" } };
    const badQuery = Kanuni.newQuery()
      .prompt(p => p.paragraph`This is a test prompt`)
      .outputText()
      .build({});
    // @ts-expect-error
    expect(() => formatter.format(badQuery)).toThrow();
  });

  it("should throw an error if a memory item maps to an invalid utterance role", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({
      roleMapper: () => "system",
    });
    // const query = makeQuery([
    //   { type: "utterance", role: "user", contents: "hi" },
    // ]);
    const query = Kanuni.newQuery()
      .prompt(p => p.paragraph`This is a test prompt`)
      .memory(m => m.utterance("user", "bob", () => "hi"))
      .outputJson(z.strictObject({
        aField: z.string().describe("A field in the output schema"),
      }))
      .build({});
    expect(() => formatter.format(query)).toThrow(/unknown utterance role/);
  });

  it("should correctly format multiple memory utterances", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
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
    expect(result.messages[1].role).toBe("user");
    expect(result.messages[1].content).toBe("hi");
    expect(result.messages[2].role).toBe("assistant");
    expect(result.messages[2].content).toBe("hello");
  });

  it("should include the correct JSON schema in response_format", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
    const query = Kanuni.newQuery()
      .prompt(p => p.paragraph`This is a test prompt`)
      .memory(m => m
        .utterance("user", () => "hi")
        .utterance("assistant", () => "hello")
      )
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

  it("should handle queries with no memory contents", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
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

  it("should handle queries with named utterances", () => {
    const formatter = new AzureOpenAIChatCompletionsJsonFormatter({});
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
});
