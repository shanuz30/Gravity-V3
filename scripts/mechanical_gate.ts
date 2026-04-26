/**
 * Mechanical Gate (TypeScript Implementation)
 */

interface ExecutionIntent {
    action: string;
    target?: string;
    query?: string;
    [key: string]: any;
}

export class MechanicalGate {
    private readonly blacklistPatterns: RegExp[] = [
        /rm\s+-rf\s+\//i,
        /mkfs/i,
        />\s*\/dev\/sda/i,
        /DROP\s+TABLE/i,
        /DELETE\s+MATCH/i
    ];

    public validateCommand(commandIntent: string): boolean {
        for (const pattern of this.blacklistPatterns) {
            if (pattern.test(commandIntent)) {
                console.error(`[MECHANICAL_GATE_BLOCK] Destructive pattern detected: ${pattern}`);
                return false;
            }
        }
        return true;
    }

    public cerebellumTranslate(intentJson: string): string {
        try {
            const data: ExecutionIntent = JSON.parse(intentJson);
            if (data.action === "query_graph" && data.query) {
                return `cypher: ${data.query}`;
            }
            return `Translated execution for: ${data.action}`;
        } catch (error) {
            console.error("[MECHANICAL_GATE_ERROR] Invalid Intent JSON structure.");
            return "";
        }
    }
}
