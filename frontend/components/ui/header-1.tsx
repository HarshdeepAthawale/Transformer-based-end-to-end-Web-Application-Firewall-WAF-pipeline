'use client';
import React from 'react';
import Link from 'next/link';
import { Button, buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { MenuToggleIcon } from '@/components/ui/menu-toggle-icon';
import { useScroll } from '@/components/ui/use-scroll';
import { createPortal } from 'react-dom';
import { Shield } from 'lucide-react';

const greenTextureDots = {
	backgroundImage: 'radial-gradient(rgba(197, 226, 70, 0.45) 1px, transparent 1px)',
	backgroundSize: '24px 24px',
} as const;

const greenTextureGradient = {
	background: 'linear-gradient(135deg, transparent 60%, rgba(197, 226, 70, 0.04) 100%)',
} as const;

export function Header({ variant = 'default' }: { variant?: 'default' | 'light' }) {
	const [open, setOpen] = React.useState(false);
	const scrolled = useScroll(10);
	const isLight = variant === 'light';

	const links = [
		{ label: 'About', href: '#about' },
		{ label: 'Products', href: '#products' },
		{ label: 'Solutions', href: '#solutions' },
		{ label: 'Customers', href: '#customers' },
		{ label: 'Pricing', href: '#pricing' },
	];

	React.useEffect(() => {
		if (open) {
			document.body.style.overflow = 'hidden';
		} else {
			document.body.style.overflow = '';
		}
		return () => {
			document.body.style.overflow = '';
		};
	}, [open]);

	return (
		<header
			className={cn(
				'sticky top-4 z-50 w-full mt-4 border-b outline-none rounded-b-lg',
				isLight
					? 'bg-white border-[#F2F2F2]'
					: cn('border-transparent', {
							'bg-background/95 supports-[backdrop-filter]:bg-background/50 border-border backdrop-blur-lg':
								scrolled,
						}),
			)}
		>
			{isLight && (
				<>
					<div
						className="absolute inset-0 pointer-events-none"
						style={greenTextureDots}
						aria-hidden
					/>
					<div
						className="absolute inset-0 pointer-events-none"
						style={greenTextureGradient}
						aria-hidden
					/>
				</>
			)}
			<nav className="relative mx-auto flex h-14 w-full max-w-5xl items-center justify-between px-4">
				<Link
					href="/"
					className={cn(
						'hover:bg-accent rounded-md p-2 flex items-center gap-2 outline-none',
						isLight && 'text-[#191A23] hover:bg-[#E8F5B8]',
					)}
					aria-label="Home"
				>
					<Shield className={cn('h-5 w-5', isLight && 'text-[#191A23]')} />
					<span className="font-bold text-lg">WAF</span>
				</Link>
				<div className="hidden items-center gap-2 md:flex">
					{links.map((link) => (
						<a
							key={link.label}
							className={buttonVariants({
								variant: 'ghost',
								className: isLight ? 'text-[#191A23] hover:bg-[#E8F5B8] hover:text-[#191A23]' : undefined,
							})}
							href={link.href}
						>
							{link.label}
						</a>
					))}
					<Button variant="outline" asChild className={isLight ? 'bg-white border-[#191A23] text-[#191A23] hover:bg-[#E8F5B8] hover:text-[#191A23]' : undefined}>
						<Link href="/login">Sign In</Link>
					</Button>
					<Button
						asChild
						className={isLight ? 'bg-[#C5E246] text-[#191A23] hover:opacity-90' : undefined}
					>
						<Link href="/dashboard">Get Started</Link>
					</Button>
				</div>
				<Button
					size="icon"
					variant="outline"
					onClick={() => setOpen(!open)}
					className={cn('md:hidden', isLight && 'border-[#191A23] text-[#191A23]')}
					aria-expanded={open}
					aria-controls="mobile-menu"
					aria-label="Toggle menu"
				>
					<MenuToggleIcon open={open} className="size-5" duration={300} />
				</Button>
			</nav>
			<MobileMenu open={open} variant={variant} className="flex flex-col justify-between gap-2">
				<div className="grid gap-y-2">
					{links.map((link) => (
						<a
							key={link.label}
							className={buttonVariants({
								variant: 'ghost',
								className: cn('justify-start', isLight && 'text-[#191A23] hover:bg-[#E8F5B8]'),
							})}
							href={link.href}
						>
							{link.label}
						</a>
					))}
				</div>
				<div className="flex flex-col gap-2">
					<Button variant="outline" className={cn('w-full', isLight && 'bg-white border-[#191A23] text-[#191A23]')} asChild>
						<Link href="/login">Sign In</Link>
					</Button>
					<Button className={cn('w-full', isLight && 'bg-[#C5E246] text-[#191A23]')} asChild>
						<Link href="/dashboard">Get Started</Link>
					</Button>
				</div>
			</MobileMenu>
		</header>
	);
}

type MobileMenuProps = React.ComponentProps<'div'> & {
	open: boolean;
	variant?: 'default' | 'light';
};

function MobileMenu({ open, variant = 'default', children, className, ...props }: MobileMenuProps) {
	if (!open || typeof window === 'undefined') return null;
	const isLight = variant === 'light';

	return createPortal(
		<div
			id="mobile-menu"
			className={cn(
				'fixed right-0 bottom-0 left-0 z-40 flex flex-col overflow-hidden border-y md:hidden',
				isLight ? 'top-[4.5rem]' : 'top-14',
				isLight
					? 'bg-white border-[#F2F2F2]'
					: 'bg-background/95 supports-[backdrop-filter]:bg-background/50 backdrop-blur-lg',
			)}
		>
			{isLight && (
				<>
					<div
						className="absolute inset-0 pointer-events-none"
						style={greenTextureDots}
						aria-hidden
					/>
					<div
						className="absolute inset-0 pointer-events-none"
						style={greenTextureGradient}
						aria-hidden
					/>
				</>
			)}
			<div
				data-slot={open ? 'open' : 'closed'}
				className={cn(
					'relative data-[slot=open]:animate-in data-[slot=open]:zoom-in-97 ease-out',
					'size-full p-4',
					className,
				)}
				{...props}
			>
				{children}
			</div>
		</div>,
		document.body,
	);
}
