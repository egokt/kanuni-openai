import type {
  Formatter,
  Query,
} from "kanuni";
import { TextualMarkdownFormatter } from "kanuni";
import { AzureOpenAI } from "openai";
import { ChatCompletionRole } from "openai/resources";

type InstructionsFormatterFunction<
  OutputSchema extends Record<string, any> = Record<string, any>,
  Role extends string = ChatCompletionRole
> = (query: Query<OutputSchema, Role>) => string;

type RoleMapperFunction<SourceRole extends string> =
  (sourceRole: SourceRole, name?: string) => ChatCompletionRole;

type InstructionsRole = 'developer' | 'system';

type AzureOpenAIChatCompletionsJsonFormatterConfig<
  OutputSchema extends Record<string, any>,
  Role extends string = ChatCompletionRole
> = {
  /**
   * Post o1 models use "developer", others use "system". Default is "system". Check which role your model uses for instructions.
   */
  instructionsRole?: InstructionsRole;

  // TODO: add model and version params so that the formatter can adapt to different OpenAI models and versions.

  /**
   * Formats the instructions for the query.
   * This is the part of the query without memory section, output, or tools.
   * 
   * The default is the format method of the TextualMarkdownFormatter in Kanuni.
   * 
   * @param query The query to format instructions for.
   * @returns The formatted instructions.
   */
  instructionsFormatter?: InstructionsFormatterFunction<OutputSchema, Role>;

  roleMapper?: RoleMapperFunction<Role>;
}

const DEFAULT_INSTRUCTIONS_ROLE: InstructionsRole = "system";

type AzureOpenAIChatCompletionsJsonFormatterParams = {};

type AzureOpenAIChatCompletionsJsonFormatterResult =
  Pick<Parameters<
    AzureOpenAI['chat']['completions']['parse']>[0],
    'messages' | 'response_format'
  >;

type Role = ChatCompletionRole;
type UtteranceRole = 'user' | 'assistant';

// This is for ensuring that the build of this package fails
// if a new role is added to the OpenAI API
const ALL_CHAT_COMPLETION_ROLES: { [key in ChatCompletionRole]: boolean } = {
  user: true,
  assistant: true,
  system: true,
  developer: true,
  function: true,
  tool: true,
};

const UTTERANCE_ROLES: { [key in ChatCompletionRole]: boolean } = {
  user: true,
  assistant: true,
  system: false,
  developer: false,
  function: false,
  tool: false,
};

function identityRoleMapper(sourceRole: ChatCompletionRole): ChatCompletionRole {
  if (ALL_CHAT_COMPLETION_ROLES[sourceRole]) {
    return sourceRole;
  }
  throw new Error(`Unknown role: ${sourceRole}`);
}

export class AzureOpenAIChatCompletionsJsonFormatter<OutputSchema extends Record<string, any>>
  implements Formatter<
    AzureOpenAIChatCompletionsJsonFormatterParams,
    AzureOpenAIChatCompletionsJsonFormatterResult,
    OutputSchema,
    Role
  >
{
  private instructionsRole: InstructionsRole;
  private instructionsFormatter: InstructionsFormatterFunction<OutputSchema, Role>;
  private roleMapper: RoleMapperFunction<Role>;

  constructor(
    {
      instructionsRole = DEFAULT_INSTRUCTIONS_ROLE,
      instructionsFormatter = (query) => new TextualMarkdownFormatter<OutputSchema, Role>().format(query),
      roleMapper = identityRoleMapper,
    }: AzureOpenAIChatCompletionsJsonFormatterConfig<OutputSchema, Role> = {},
  ) {
    this.instructionsRole = instructionsRole;
    this.instructionsFormatter = instructionsFormatter;
    this.roleMapper = roleMapper;
 }

  format(
    query: Query<OutputSchema, Role>,
    params: AzureOpenAIChatCompletionsJsonFormatterParams = {},
  ): AzureOpenAIChatCompletionsJsonFormatterResult {
    const memoryItems = this.formatMemoryItems(query, params);
    const responseJsonSchema = this.formatJsonSchema(query, params);

    return {
      messages: memoryItems,
      response_format: responseJsonSchema,
    };
  }

  // Warn: This method only supports utterances in the memory section.
  // TODO: Extend this to support other types of memory items, i.e. tools, when they are implemented.
  private formatMemoryItems(
    query: Query<OutputSchema, Role>,
    _params: AzureOpenAIChatCompletionsJsonFormatterParams,
  ): AzureOpenAIChatCompletionsJsonFormatterResult['messages'] {
    const instructionsRole = this.instructionsRole;
    const instructions = this.instructionsFormatter(query);

    const memoryItems = [
      {
        role: instructionsRole,
        content: instructions,
      },
      ...(query.memory?.contents || []).map((item) => {
        switch (item.type) {
          case "utterance":
            const role = this.roleMapper(item.role, item.name);
            if (!UTTERANCE_ROLES[role]) {
              throw new Error(`Role mapping for '${item.role}' resulted in unknown utterance role: ${role}`);
            }
            return {
              role: role as UtteranceRole, // this type casting is safe due to the conditional above
              ...(item.name !== undefined && item.name !== '' ? { name: item.name } : {}),
              content: item.contents,
            };
          default:
            throw new Error(`Unknown memory item type: ${item.type}`);
        }
      }),
    ];

    return memoryItems;
  }

  private formatJsonSchema(
    query: Query<OutputSchema, Role>,
    params: AzureOpenAIChatCompletionsJsonFormatterParams,
  ): AzureOpenAIChatCompletionsJsonFormatterResult['response_format'] {
    // This method should format the JSON schema based on the query and params.
    // For now, we return an empty object as a placeholder.
    query;
    params;
    return {
      // TODO: Support 'json_objet' response format for models that do not
      // support 'json_schema', or if the caller explicitly requests it.
      type: 'json_schema',

      json_schema: {
        name: 'response_schema',
        
        // FIXME: This is a placeholder schema. Replace with actual schema logic.
        description: 'Schema for the response',
      }
    };
  }

}
