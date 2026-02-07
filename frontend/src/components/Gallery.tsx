import { SimpleGrid, Image, Paper, Text, Stack } from '@mantine/core';
import { useStore } from '../store/useStore';

export function Gallery() {
    const { gallery, setCurrentImage } = useStore();

    if (gallery.length === 0) {
        return (
            <Stack mt="xl">
                <Text size="lg" fw={500}>Recent Generations</Text>
                <Text c="dimmed" size="sm">No images generated yet.</Text>
            </Stack>
        );
    }

    return (
        <Stack mt="xl">
            <Text size="lg" fw={500}>Recent Generations</Text>
            <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5 }}>
                {gallery.map((img, i) => (
                    <Paper key={i} shadow="xs" p={5} withBorder style={{ cursor: 'pointer', overflow: 'hidden' }} onClick={() => setCurrentImage(img)}>
                        <Image
                            src={img}
                            h={150}
                            fit="cover"
                            radius="sm"
                        />
                    </Paper>
                ))}
            </SimpleGrid>
        </Stack>
    );
}
