import { ActionIcon, AppShell, Burger, Group, Stack, Text, useMantineColorScheme } from '@mantine/core';
import { useDisclosure } from '@mantine/hooks';
import { IconBolt, IconMoon, IconSun } from '@tabler/icons-react';
import { GenerationSettings } from './components/GenerationSettings';
import { ImagePreview } from './components/ImagePreview';
import { Gallery } from './components/Gallery';
import { useStore } from './store/useStore';

export default function App() {
  const [opened, { toggle }] = useDisclosure();
  const { serverStatus } = useStore();
  const { colorScheme, toggleColorScheme } = useMantineColorScheme();
  const dark = colorScheme === 'dark';

  return (
    <AppShell
      header={{ height: 60 }}
      navbar={{
        width: 350,
        breakpoint: 'sm',
        collapsed: { mobile: !opened },
      }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Group>
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <IconBolt size={24} color={dark ? '#ffd43b' : '#228be6'} />
            <Text size="xl" fw={700} gradient={{ from: 'blue', to: 'cyan' }} variant="gradient">
              LightDiffusion Next
            </Text>
          </Group>

          <Group>
            {serverStatus ?
              <Text size="xs" c="green" fw={500}>● Connected</Text> :
              <Text size="xs" c="red" fw={500}>● Disconnected</Text>
            }
            <ActionIcon
              variant="default"
              color={dark ? 'yellow' : 'blue'}
              onClick={() => toggleColorScheme()}
              title="Toggle color scheme"
              size="lg"
            >
              {dark ? <IconSun size={18} /> : <IconMoon size={18} />}
            </ActionIcon>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md" style={{ overflowY: 'auto' }}>
        <GenerationSettings />
      </AppShell.Navbar>

      <AppShell.Main>
        <Stack gap="lg">
          <ImagePreview />
          <Gallery />
        </Stack>
      </AppShell.Main>
    </AppShell>
  );
}
