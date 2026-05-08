import { MechanicalGate } from '../scripts/mechanical_gate';

describe('MechanicalGate', () => {
    let gate: MechanicalGate;

    beforeEach(() => {
        gate = new MechanicalGate();
    });

    describe('validateCommand', () => {
        it('should return true for safe commands', () => {
            expect(gate.validateCommand('ls -la')).toBe(true);
            expect(gate.validateCommand('echo "hello"')).toBe(true);
            expect(gate.validateCommand('SELECT * FROM users')).toBe(true);
        });

        it('should return false for unsafe commands', () => {
            expect(gate.validateCommand('rm -rf /')).toBe(false);
            expect(gate.validateCommand('rm -rf /etc')).toBe(false);
            expect(gate.validateCommand('mkfs.ext4 /dev/sda1')).toBe(false);
            expect(gate.validateCommand('echo "bad" > /dev/sda')).toBe(false);
            expect(gate.validateCommand('DROP TABLE users;')).toBe(false);
            expect(gate.validateCommand('DELETE MATCH p=() RETURN p;')).toBe(false);
        });
    });

    describe('cerebellumTranslate', () => {
        it('should translate query_graph action correctly', () => {
            const intent = JSON.stringify({
                action: 'query_graph',
                query: 'MATCH (n) RETURN n'
            });
            expect(gate.cerebellumTranslate(intent)).toBe('cypher: MATCH (n) RETURN n');
        });

        it('should translate other actions generically', () => {
            const intent = JSON.stringify({
                action: 'do_something_else'
            });
            expect(gate.cerebellumTranslate(intent)).toBe('Translated execution for: do_something_else');
        });

        it('should handle invalid JSON safely', () => {
            const invalidIntent = 'not a valid json';
            expect(gate.cerebellumTranslate(invalidIntent)).toBe('');
        });
    });
});
